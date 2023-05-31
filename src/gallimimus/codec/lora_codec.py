from __future__ import annotations

import jax
import flax
import jax.numpy as jnp
import flax.linen as nn

from typing import Tuple, Callable, List

from gallimimus.codec.abstract_codec import Codec, Embedding

CategoricalObservation = jax.Array  # of shape () and dtype int
CategoricalContext = None
CategoricalPrediction = jax.Array  # un-normalized logits of shape (vocab_size,)

from flax.core.scope import VariableDict
from jax.random import KeyArray

FilterFunction = Callable[[List[str], jax.Array], bool]


def init_lora_params(
    rng: KeyArray,
    params: VariableDict,
    filter_fn: FilterFunction,
    r: int,
) -> VariableDict:
    """Initialize the PyTree containing the low-rank perturbations.

    :param rng: a jax PRNGKey
    :param params: the PyTree of the pretrained parameters
    :param filter_fn: A function of signature ``(param_name: t.List[str], param: jax.Array) -> bool`` deciding if a parameter is finetuned.
    :param r: The rank of the LoRA approximation.
    :return: A PyTree which is a subset of ``pretrained_params`` and contains the initialized low-rank perturbations
    """
    params_flat = flax.traverse_util.flatten_dict(
        params,
    )

    rngs = jax.random.split(key=rng, num=len(params_flat))

    def _init_lora_params(param, rng):
        shape = param.shape
        assert len(shape) == 2

        a, b = shape[0], shape[-1]
        A = jax.random.normal(key=rng, shape=(a, r))
        B = jnp.zeros(shape=(r, b))
        return A, B

    lora_params_flat = {
        param_name: _init_lora_params(param, rng)
        for rng, (param_name, param) in zip(rngs, params_flat.items())
        if filter_fn(param_name, param)
    }
    lora_params = flax.traverse_util.unflatten_dict(lora_params_flat)
    return lora_params


def lora_combine_vars(
    pretrained_var: jax.Array,
    lora_vars: Tuple[jax.Array, jax.Array],
    alpha: float = 1.0,
):
    A, B = lora_vars
    return pretrained_var + alpha * A @ B


def lora_combine_params(
    pretrained_params: VariableDict,
    lora_params: VariableDict,
    alpha: float = 1.0,
) -> VariableDict:
    """Adds the LoRA fine-tuning to the pretrained parameters.

    :param pretrained_params: A PyTree containing the original parameters
    :param lora_params: A PyTree which is a subset of ``pretrained_params`` and contains the trained low-rank perturbations to add
    :param alpha: scaling of the perturbation in the sum
    :return: A PyTree containing the updated parameters (same shape as ``pretrained_params``)
    """
    lora_params_flat = flax.traverse_util.flatten_dict(
        lora_params,
    )
    pretrained_params_flat = flax.traverse_util.flatten_dict(pretrained_params)

    total_flat = {
        k: (
            lora_combine_vars(pretrained_param, lora_params_flat[k], alpha)
            if k in lora_params_flat
            else pretrained_param
        )
        for k, pretrained_param in pretrained_params_flat.items()
    }

    return flax.traverse_util.unflatten_dict(total_flat)


class LoraCodec(Codec):
    """Handles an integer in [0, ``vocab_size``-1].

    An observation is a jax.Array of shape ``()`` containing an ``int`` in [0, ``vocab_size``-1].

    :param embed_dim: size of the embeddings.
    :param vocab_size: Number of possible values"""

    subcodec_in: Codec
    lora_module_name: str
    filter_fn: FilterFunction
    r: int

    def setup(self):
        self.subcodec = self.subcodec_in.clone()

    @nn.compact
    def apply_lora(self, shared_dicts):
        model_dict, params_dict = shared_dicts

        pretrained_params = params_dict[self.lora_module_name]
        lora_params = self.param(
            "lora_params", init_lora_params, pretrained_params, self.filter_fn, self.r
        )

        summed_params = lora_combine_params(
            pretrained_params=pretrained_params,
            lora_params=lora_params,
        )

        updated_params_dict = params_dict.copy({self.lora_module_name: summed_params})
        return model_dict, updated_params_dict

    def encode(
        self, x: CategoricalObservation, shared_dicts
    ) -> Tuple[Embedding, CategoricalContext]:
        lora_shared_dicts = self.apply_lora(shared_dicts)
        return self.subcodec.encode(x=x, shared_dicts=lora_shared_dicts)

    def decode(
        self, conditioning_vector: Embedding, context: CategoricalContext, shared_dicts
    ) -> CategoricalPrediction:
        lora_shared_dicts = self.apply_lora(shared_dicts)
        return self.subcodec.decode(
            conditioning_vector=conditioning_vector,
            context=context,
            shared_dicts=lora_shared_dicts,
        )

    def sample(
        self, conditioning_vector: Embedding, shared_dicts
    ) -> Tuple[CategoricalObservation, Embedding]:
        lora_shared_dicts = self.apply_lora(shared_dicts)
        return self.subcodec.sample(
            conditioning_vector=conditioning_vector, shared_dicts=lora_shared_dicts
        )

    def loss(
        self, x: CategoricalObservation, prediction: CategoricalPrediction, shared_dicts
    ) -> jnp.ndarray:
        lora_shared_dicts = self.apply_lora(shared_dicts)
        return self.subcodec.loss(
            x=x, prediction=prediction, shared_dicts=lora_shared_dicts
        )

    def example(self, shared_dicts):
        return self.subcodec_in.example(shared_dicts=shared_dicts)

    def init_pass(self):
        return super(LoraCodec, self).init_pass()
