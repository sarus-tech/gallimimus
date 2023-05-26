from __future__ import annotations

import jax

import jax.numpy as jnp
import flax.linen as nn

from typing import Tuple

from gallimimus.codec.abstract_codec import Codec, Embedding

CategoricalObservation = jax.Array  # of shape () and dtype int
CategoricalContext = None
CategoricalPrediction = jax.Array  # un-normalized logits of shape (vocab_size,)


class CategoricalCodec(Codec):
    """Handles an integer in [0, ``vocab_size``-1].

    An observation is a jax.Array of shape ``()`` containing an ``int`` in [0, ``vocab_size``-1].

    :param embed_dim: size of the embeddings.
    :param vocab_size: Number of possible values"""

    vocab_size: int

    def setup(self):
        self.embedder = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
            embedding_init=nn.zeros,
        )
        self.bias = self.param("bias_decoding", nn.ones, (self.vocab_size,))

    def encode(self, x: CategoricalObservation) -> Tuple[Embedding, CategoricalContext]:
        assert x.shape == ()
        embedding = self.embedder(inputs=x)
        return embedding, None

    def decode(
        self, conditioning_vector: Embedding, context: CategoricalContext
    ) -> CategoricalPrediction:
        prediction = self.embedder.attend(query=conditioning_vector) + self.bias
        return prediction

    def sample(
        self, conditioning_vector: Embedding
    ) -> Tuple[CategoricalObservation, Embedding]:
        assert conditioning_vector.shape == (self.embed_dim,)
        prediction = self.decode(conditioning_vector=conditioning_vector, context=None)
        rng = self.make_rng(name="sample")
        sample = jax.random.categorical(
            key=rng,
            logits=prediction,
        )

        embedding, _ = self.encode(x=sample)
        return sample, embedding

    def loss(
        self, x: CategoricalObservation, prediction: CategoricalPrediction
    ) -> jnp.ndarray:
        logits_normalized = jax.nn.log_softmax(x=prediction)
        loss_x = -(
            logits_normalized * jax.nn.one_hot(x=x, num_classes=self.vocab_size)
        ).sum()
        return loss_x

    def example(self):
        return jnp.array(self.vocab_size - 1)
