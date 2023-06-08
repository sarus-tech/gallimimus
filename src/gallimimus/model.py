"""A genererative model for complex and hierarchical data"""
from __future__ import annotations

from typing import Tuple, Any, Dict

import flax.traverse_util
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.scope import VariableDict
from jax.random import PRNGKeyArray

from gallimimus.codec.abstract_codec import Codec, Observation
from gallimimus.shared_codecs import SharedCodecs, init_shared_trained_param_dict

ModelDict = flax.core.FrozenDict[str, Codec]
ParamDict = flax.core.FrozenDict[str, VariableDict]


class UnitMetaLearner(nn.Module):
    """The ``codec`` is embedded in a ``Metalearner`` which holds an additionnal ``starting_embedding`` parameter.
    The ``starting_embedding`` is the conditioning vector from which the autoregressive sampling starts.

    This module works with single observations (not batches). In practice, the model is vmapped over in the following ``BatchMetaLearner``.
    """

    codec_in: str
    model_dict: ModelDict
    pretrained_params_dict: ParamDict

    def setup(self):
        embed_dim = self.model_dict[self.codec_in].embed_dim
        self.starting_embedding = self.param(
            "starting_embedding", nn.ones, (embed_dim,)
        )

        self.trained_params_dict = self.param(
            "trained_params_dict",
            init_shared_trained_param_dict,
            self.model_dict,
            self.pretrained_params_dict,
        )

    def _shared_codecs(self):
        params_dict = self.pretrained_params_dict.copy(
            add_or_replace=self.trained_params_dict
        )
        shared_codecs = SharedCodecs(
            shared_models_dict=self.model_dict, shared_params_dict=params_dict
        )
        return shared_codecs

    def __call__(self, x: Observation):
        """Compute the negative log-likelihood of sampling ``x`` from the model.

        We want to find the likelihood of sampling ``x`` starting from the ``starting_embedding``.
        Sampling is autoregressive for complex structures, so the probability of ``x`` is the product of the probabilities of
        sampling each ``x_i`` conditioned on having sampled ``x_1, ..., x_(i-1)``. Therefore we need the intermediate embeddings
        ``embed(x1, ..., x_(i-1))`` to compute the probability of sampling each ``x_i``.

        :param x: An observation.
        :return: The negative log-likelihood of ``x`` in the distribution generated by the model.
        """

        shared_codecs = self._shared_codecs()
        # Hence the computation is done in two steps:
        # 1/ compute the embeddings of the substructures of x. They are returned in the `context` variable:
        embedding, context = shared_codecs.encode(model_name=self.codec_in, x=x)

        # 2/ predict the next column starting from the `starting_embedding` and as-if `x` was autoregressively sampled:
        prediction = shared_codecs.decode(
            model_name=self.codec_in,
            conditioning_vector=self.starting_embedding,
            context=context,
        )

        # then we can evaluate P(x) = pred1(x1) x pred2(x2) x ...
        loss_tree_x = shared_codecs.loss(
            model_name=self.codec_in, x=x, prediction=prediction
        )
        loss_x = jnp.array(jax.tree_util.tree_flatten(loss_tree_x)[0]).mean()

        return loss_x

    def sample(self):
        """:return: A sample from the distribution generated by the model."""

        rng = self.make_rng("sample")
        shared_codecs = self._shared_codecs()

        sample = shared_codecs.sample(
            model_name=self.codec_in,
            conditioning_vector=self.starting_embedding,
            rng=rng,
        )
        return sample

    def example(self):
        """:return: An example observation of the data type expected by the model."""
        shared_codecs = self._shared_codecs()
        return self.model_dict[self.codec_in].example(shared_codecs)

    def no_method(self):
        pass


class MetaLearner:
    """The MetaLearner is the complete generative model. The type of the data it generates is defined by the
    provided ``Codec``.

    :param codec_in: The codec corresponding to the data type generated by the model.
    :param model_dict: A collection of codecs that can be shared in the type-tree
    :param params_dict: A collection of parameters for the shared codecs. For each model in ``model_dict``:

        - If parameters are provided, they are saved and used as-is, without being trained. It is still possible to finetune them using the :class:`gallimimus.codec.LoraCodec`
        - If the parameters are not provided, they are initialized as parameters of the MetaLearner, and then trained.
    """

    def __init__(
        self,
        codec_in: str,
        model_dict: Dict[str, Codec],
        pretrained_params_dict: Dict[str, VariableDict],
    ):
        # the MetaLearner is created by vmapping the methods of the UnitMetaLearner defined above.
        # `unit_metalearner` is a stateful flax module, we convert all the methods we need to pure jax functions before vmapping:

        model_dict = flax.core.frozen_dict.freeze(model_dict)
        pretrained_params_dict = flax.core.frozen_dict.freeze(pretrained_params_dict)
        unit_metalearner = UnitMetaLearner(
            codec_in=codec_in,
            model_dict=model_dict,
            pretrained_params_dict=pretrained_params_dict,
            parent=None,
        )

        self.init_fun = lambda rng: unit_metalearner.init(
            rngs={"params": rng}, method=unit_metalearner.no_method
        )

        self.apply_fun = lambda params, x: unit_metalearner.apply(
            variables={"params": params},
            x=x,
        )

        self.sample_fun = lambda params, rng: unit_metalearner.apply(
            variables={"params": params},
            rngs={"sample": rng},
            method="sample",
        )

        self.example = lambda _: unit_metalearner.apply(
            variables={},
            method="example",
        )

    def init(self, rng: PRNGKeyArray) -> VariableDict:
        """Initalize the parameters of the model.

        :param rng: A Jax PRNGKey.
        :return: A Pytree of initial parameters."""
        # x is a single observation (and not a batch), the example provided by the Codecs can be used
        params = self.init_fun(rng=rng)["params"]
        return params

    def sample(self, params: VariableDict, rng: PRNGKeyArray, size: int):
        """Sample a batch of chosen size from the distribution represented by the model with the provided parameters.

        :param params: A Pytree of parameters.
        :param rng: A Jax PRNGKey.
        :param size: The number of requested samples.
        :return: A batch of samples of chosen size, stacked on the first dimension of the leaves.
        """
        vmapped_sample_fun = jax.vmap(self.sample_fun, in_axes=(None, 0))

        rngs = jax.random.split(rng, size)
        samples, embeddings = vmapped_sample_fun(params, rngs)
        return samples

    
    def batch_loss(
        self, params: VariableDict, xs: Observation
    ) -> Tuple[jnp.ndarray, Any]:
        """For a batch of observations ``xs``, compute the average loss (the average NLL of ``x`` in the distribution represented
        by ``params``) and its associated gradient.

        :param params: A Pytree of parameters.
        :param xs: A batch of observations, stacked on the first dimension of the leaves.
        :return: A tuple containing the average loss of the batch (an array of shape ``()``) and its gradient with respect to the parameters
            (a Pytree of the same shape as ``params``)."""
        vmapped_apply_fun = jax.vmap(self.apply_fun, in_axes=(None, 0))
        loss_batch = vmapped_apply_fun(params, xs)
        loss = loss_batch.mean()
        return loss
    
    def loss_and_grad(
        self, params: VariableDict, xs: Observation
    ) -> Tuple[jnp.ndarray, Any]:
        """For a batch of observations ``xs``, compute the average loss (the average NLL of ``x`` in the distribution represented
        by ``params``) and its associated gradient.

        :param params: A Pytree of parameters.
        :param xs: A batch of observations, stacked on the first dimension of the leaves.
        :return: A tuple containing the average loss of the batch (an array of shape ``()``) and its gradient with respect to the parameters
            (a Pytree of the same shape as ``params``)."""
        # vmap first, average the losses, then grad to obtain batch-level gradient
        grad_vmapped_apply_fun = jax.value_and_grad(fun=self.batch_loss)
        batch_loss, batch_grad = grad_vmapped_apply_fun(params, xs)
        return batch_loss, batch_grad

    def loss_and_per_example_grad(
        self, params: VariableDict, xs: Observation
    ) -> Tuple[jnp.ndarray, Any]:
        """For a batch of observations ``xs``, compute the average loss and *per-instance* gradients
        (per-instance gradients are required for DP-SGD, so that they can be clipped).

        :param params: A Pytree of parameters.
        :param xs: A batch of observations, stacked on the first dimension of the leaves.
        :return: A tuple containing the average loss of the batch (an array of shape ``()``) and the *per-instance* gradient with respect to the parameters
            (a stack of Pytrees of the same shape as ``params``, stacked on the first dimension of the leaves).
        """
        # grad then vmap to obtain instance-level gradients
        grad_apply_fun = jax.value_and_grad(fun=self.apply_fun)
        vmapped_grad_apply_fun = jax.vmap(grad_apply_fun, in_axes=(None, 0))

        per_ex_loss, per_ex_grad = vmapped_grad_apply_fun(params, xs)
        return per_ex_loss.mean(), per_ex_grad
