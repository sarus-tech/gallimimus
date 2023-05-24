import jax
import jax.numpy as jnp
import flax.linen as nn

import typing as t

from gallimimus.model import Embedding, Codec


CategoricalObservation = jax.Array  # of shape () and dtype int
CategoricalContext = None
CategoricalPrediction = jax.Array  # un-normalized logits of shape (vocab_size,)


class CategoricalCodec(Codec):
    """Handles an integer in [0, ``vocab_size``-1]."""

    vocab_size: int
    """Number of possible values."""

    def setup(self):
        self.embedder = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
            embedding_init=nn.zeros,
        )
        self.bias = self.param("bias_decoding", nn.ones, (self.vocab_size,))

    def encode(
        self, x: CategoricalObservation
    ) -> t.Tuple[Embedding, CategoricalContext]:
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
    ) -> t.Tuple[CategoricalObservation, Embedding]:
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