from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from gallimimus.codec.abstract_codec import Codec, Embedding
from gallimimus.shared_codecs import SharedCodecs

CategoricalObservation = jax.Array  # of shape () and dtype int
CategoricalContext = None
CategoricalPrediction = jax.Array  # un-normalized logits of shape (vocab_size,)


class CategoricalCodec(Codec):
    """Handles an integer in [0, ``vocab_size``-1].

    An observation is a jax.Array of shape ``()`` containing an ``int`` in
    the interval [0, ``vocab_size``-1].

    :param embed_dim: size of the embeddings.
    :param vocab_size: Number of possible values
    """

    vocab_size: int

    def setup(self):
        self.embedder = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
        )
        self.bias = self.param("bias_decoding", nn.zeros, (self.vocab_size,))

    def encode(
        self, x: CategoricalObservation, shared_codecs: SharedCodecs
    ) -> Tuple[Embedding, CategoricalContext]:
        assert x.shape == ()
        embedding = self.embedder(inputs=x)
        return embedding, None

    def decode(
        self,
        conditioning_vector: Embedding,
        context: CategoricalContext,
        shared_codecs: SharedCodecs,
    ) -> CategoricalPrediction:
        prediction = self.embedder.attend(query=conditioning_vector) + self.bias
        return prediction

    def sample(
        self, conditioning_vector: Embedding, shared_codecs: SharedCodecs
    ) -> Tuple[CategoricalObservation, Embedding]:
        assert conditioning_vector.shape == (self.embed_dim,)
        prediction = self.decode(
            conditioning_vector=conditioning_vector,
            context=None,
            shared_codecs=shared_codecs,
        )
        rng = self.make_rng(name="sample")
        sample = jax.random.categorical(
            key=rng,
            logits=prediction,
        )

        embedding, _ = self.encode(x=sample, shared_codecs=shared_codecs)
        return sample, embedding

    def loss(
        self,
        x: CategoricalObservation,
        prediction: CategoricalPrediction,
        shared_codecs: SharedCodecs,
    ) -> jnp.ndarray:
        logits_normalized = jax.nn.log_softmax(x=prediction)
        loss_x = -(
            logits_normalized * jax.nn.one_hot(x=x, num_classes=self.vocab_size)
        ).sum()
        return loss_x

    def example(self, shared_codecs: SharedCodecs):
        return jnp.array(self.vocab_size - 1)
