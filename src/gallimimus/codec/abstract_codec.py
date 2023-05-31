from __future__ import annotations

import abc
from typing import TypeVar, Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp

Embedding = jax.Array  # of size `embed_dim`
Observation = TypeVar("Observation")
Context = TypeVar("Context")
Prediction = TypeVar("Prediction")


class Codec(nn.Module, abc.ABC):
    """The abstract interface of a codec. Each concrete codec implements the ``Codec`` interface for a specific data-type."""

    embed_dim: int
    """Size of the embeddings. Should be the same for all the codecs in the tree of codecs."""

    @abc.abstractmethod
    def encode(self, x: Observation, shared_dicts) -> Tuple[Embedding, Context]:
        """Encode an observation.

        :param shared_dicts:
        :param x: An observation of the data type expected by the codec.
        :return: A pair containing

         - an embedding vector of shape ``(self.embed_dim,)``
         - a context containing the embeddings for the substructures of ``x``."""
        ...

    @abc.abstractmethod
    def decode(
        self, conditioning_vector: Embedding, context: Context, shared_dicts
    ) -> Prediction:
        """Turn a ``conditioning_vector`` into a predicted probability distribution,
        using the embeddings in the ``context`` in places where autoregressive sampling would occur.

        :param shared_dicts:
        :param conditioning_vector: Conditioning vector of shape ``(self.embed_dim,)``.
        :param context: Embeddings of the substructures as given by ``encode``.
        :return: A representation of the probability distribution predicted from the conditioning vector.
        """
        ...

    @abc.abstractmethod
    def sample(
        self, conditioning_vector: Embedding, shared_dicts
    ) -> Tuple[Observation, Embedding]:
        """
        Sample a single observation, as conditioned by the ``conditioning_vector``.

        :param shared_dicts:
        :param conditioning_vector: Conditioning vector of shape ``(self.embed_dim,)``.
        :return: A sample from the probability predicted by the conditioning vector, and its embedding.
        """
        ...

    @abc.abstractmethod
    def loss(
        self, x: Observation, prediction: Prediction, shared_dicts
    ) -> jnp.ndarray:  # of shape ()
        """Returns the negative log-likelihood of ``x`` in the provided distribution.

        :param shared_dicts:
        :param x: An observation of the data type expected by the codec.
        :param prediction: A representation of a probability distribution.
        :return: The negative log-likelihood of ``x`` in this distribution."""
        ...

    @abc.abstractmethod
    def example(self, shared_dicts) -> Observation:
        """Convenience function which provides an example input for the model.

        :return: An example observation of the data type expected by the codec."""
        ...

    def init_pass(self):
        # only used to initialize shared codecs
        shared_dicts = ({}, {})
        x = self.example(shared_dicts)
        embedding, context = self.encode(x=x, shared_dicts=shared_dicts)
        prediction = self.decode(
            conditioning_vector=embedding, context=context, shared_dicts=shared_dicts
        )
        loss_x = self.loss(x=x, prediction=prediction, shared_dicts=shared_dicts)
        return loss_x
