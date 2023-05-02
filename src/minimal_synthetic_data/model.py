import abc
import typing as t

import jax
from flax import linen as nn


Embedding = jax.Array  # of size `embed_dim`

Observation = t.TypeVar("Observation")
Context = t.TypeVar("Context")
Prediction = t.TypeVar("Prediction")


class Codec(nn.Module, abc.ABC):
    """the abstract interface of a codec"""

    embed_dim: int

    @abc.abstractmethod
    def encode(self, x: Observation) -> t.Tuple[Embedding, Context]:
        """Turn an observation `x` into
        - an embedding vector of fixed size `(self.embed_dim,)`
        - a context containing the embeddings for the substructures of x"""
        ...

    @abc.abstractmethod
    def decode(self, conditioning_vector: Embedding, context: Context) -> Prediction:
        """Turn a `conditioning_vector` into a predicted probability distribution,
        using the embeddings in the `context` in places where autoregressive sampling would occur.
        """
        ...

    @abc.abstractmethod
    def sample(self, conditioning_vector: Embedding) -> t.Tuple[Observation, Embedding]:
        """Sample a single observation, as conditioned by the `conditioning_vector`."""
        ...

    @abc.abstractmethod
    def loss(self, x: Observation, prediction: Prediction) -> jax.Array:  # of shape ()
        """The `prediction` represents a probability distribution.
        Returns the negative log-likelyhood of `x` in this distribution."""
        ...

    @abc.abstractmethod
    def example(self) -> Observation:
        """Convenience function which provides an example input for the model."""
        ...


class MetaLearner(nn.Module):
    """The `codec` is embedded in a `Metalearner` which holds an additionnal `starting_embedding` parameter.
    The `starting_embedding` is the conditioning vector from which the autoregressive sampling starts.

    This module works with single observations (not batches). The concrete model used in practice is the following BatchMetaLearner.
    """

    codec: Codec

    def setup(self):
        self.starting_embedding = self.param(
            "starting_embedding", nn.zeros, (self.codec.embed_dim,)
        )

    def full_pass(self, x: Observation):
        """Compute the negative log-likelyhood of sampling `x` from the model.

        We want to find the likelyhood of sampling `x` starting from the `starting_embedding`.
        Sampling is autoregressive for complex structures, so the probability of x is the product of the probabilites of
        sampling each x_i conditioned on having sampled x_1,...,x_(i-1). To compute the probability of sampling
        x_i at each step, we need the intermediate embeddings embed(x1,...,x_(i-1))

        Hence the computation is done in two steps:"""

        # 1/ compute the embeddings of the substructures of x. They are returned in the `context` variable:
        embedding, context = self.codec.encode(x=x)

        # 2/ predict the next column starting from the `starting_embedding` and as-if `x` was autoregressively sampled
        prediction = self.codec.decode(
            conditioning_vector=self.starting_embedding, context=context
        )

        # then we can evaluate P(x) = pred1(x1) x pred2(x2) x ...
        loss_x = self.codec.loss(x=x, prediction=prediction)
        return loss_x

    def sample(self):
        sample = self.codec.sample(conditioning_vector=self.starting_embedding)
        return sample

    def example(self):
        return self.codec.example()


class BatchMetaLearner:
    """The necessary methods to initialize, train, and sample from the model."""

    def __init__(self, codec: Codec):
        # `metalearner` is a stateful flax module, we convert all the methods we need to pure jax functions:
        metalearner = MetaLearner(codec=codec, parent=None)

        self.init_fun = lambda rng, x: metalearner.init(
            rngs={"params": rng}, x=x, method="full_pass"
        )

        self.apply_fun = lambda params, x: metalearner.apply(
            variables={"params": params},
            x=x,
            method="full_pass",
        )

        self.sample_fun = lambda params, rng: metalearner.apply(
            variables={"params": params},
            rngs={"sample": rng},
            method="sample",
        )

        self.example = metalearner.example()

    def init(self, rng):
        """Initalize the parameters of the model."""
        # x is a single observation (and not a batch), the example provided by the Codecs can be used
        params = self.init_fun(rng=rng, x=self.example)["params"]
        return params

    def loss_and_per_example_grad(self, params, xs):
        """For a batch of observations `xs`, compute the *per-instance* losses and their associated gradients
        (per-instance gradients are required for DP-SGD to be able to clip them)."""
        # grad then vmap to obtain instance-level gradients
        grad_apply_fun = jax.value_and_grad(fun=self.apply_fun)
        vmapped_grad_apply_fun = jax.vmap(grad_apply_fun, in_axes=(None, 0))

        per_ex_loss, per_ex_grad = vmapped_grad_apply_fun(params, xs)
        return per_ex_loss.mean(), per_ex_grad

    def loss_and_grad(self, params, xs):
        """For a batch of observations `xs`, compute the batched loss and its associated gradient."""
        # vmap first, average the losses, then grad to obtain batch-level gradient
        vmapped_apply_fun = jax.vmap(self.apply_fun, in_axes=(None, 0))
        scalar_apply_fun = lambda params, xs: vmapped_apply_fun(params, xs).mean()
        grad_vmapped_apply_fun = jax.value_and_grad(fun=scalar_apply_fun)

        batch_loss, batch_grad = grad_vmapped_apply_fun(params, xs)
        return batch_loss, batch_grad

    def sample(self, params, rng, size):
        vmapped_sample_fun = jax.vmap(self.sample_fun, in_axes=(None, 0))

        rngs = jax.random.split(rng, size)
        samples, embeddings = vmapped_sample_fun(params, rngs)
        return samples