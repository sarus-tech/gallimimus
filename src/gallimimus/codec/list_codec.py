from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from gallimimus.codec.abstract_codec import (
    Codec,
    Embedding,
    Observation,
    Context,
    Prediction,
)
from gallimimus.codec.categorical_codec import CategoricalCodec
from gallimimus.shared_codecs import SharedCodecs
from gallimimus.transformer import Transformer

# For a ListCodec of buffer_size N:
ListObservation = Tuple[jax.Array, Observation]
# real length (array of shape ()), and stacked Pytrees of sub-observations
ListContext = Tuple[jax.Array, Context]
# encoded_embeddings, stacked Pytrees of sub-contexts
ListPrediction = Tuple[jax.Array, Prediction]
# length logits, stacked Pytrees of sub-predictions


class ListCodec(Codec):
    """Formally the ListCodec behaves like a ``StructCodec[Categorical(max_len+1), subcodec, ..., subcodec]`` where ``subcodec``
    is repeated ``buffer_size`` times, except that the loss of ``(len_x, items_x)`` only looks at the ``len_x`` first columns.

    For efficiency reasons, a vectorized version of the ``subcodec`` is used (otherwise jit compilation unrolls the loops).
    Due to this, an observation is a Pytree where the items are stacked on the first dimension of their leaves.

    If the observations of the ``subcodec`` are of type SubObservation, observations for the ``ListCodec``
    are a tuple containing:

    - a jax.Array of shape ``()`` containing an ``int`` in [0, ``max_len``]
    - a stack of SubObservations (which is a PyTree), so that length of the first dimension of the leaves is ``buffer_size``

    :param embed_dim: Size of the embeddings.
    :param subcodec_in: Codec used to generate the items in the list.
    :param n_heads: Number of transformer heads.
    :param n_blocks: Number of transformer blocks.
    :param max_len: Maximum size of the generated list.
    :param buffer_size: Size of the buffer used for training. Must be smaller than ``max_len``.
    """

    subcodec_in: str

    n_heads: int
    n_blocks: int
    max_len: int
    buffer_size: int

    def setup(self):
        self.len_codec = CategoricalCodec(
            embed_dim=self.embed_dim,
            vocab_size=self.max_len + 1,
        )

        self.encoder = Transformer(
            num_heads=self.n_heads, num_blocks=self.n_blocks, embed_dim=self.embed_dim
        )
        self.decoder = Transformer(
            num_heads=self.n_heads, num_blocks=self.n_blocks, embed_dim=self.embed_dim
        )

    def encode(
        self, x: ListObservation, shared_codecs: SharedCodecs
    ) -> Tuple[Embedding, ListContext]:
        x_len, x_items = x

        # encode the length and items independently
        embedding_len, _ = self.len_codec.encode(x=x_len, shared_codecs=shared_codecs)

        embedding_items, subcontexts = shared_codecs.encode(
            model_name=self.subcodec_in, x=x_items, vmapped=True
        )

        # transform the embeddings with the encoder
        embeddings = jnp.vstack([embedding_len, embedding_items])  # length N + 1
        encoded_embeddings = self.encoder(inputs=embeddings)

        return encoded_embeddings[-1], (encoded_embeddings, subcontexts)

    def decode(
        self,
        conditioning_vector: Embedding,
        context: ListContext,
        shared_codecs: SharedCodecs,
    ) -> ListPrediction:
        encoded_embeddings, subcontexts = context

        # transform the *conditioned* embeddings with the decoder
        conditioning_vectors = jnp.vstack(
            [conditioning_vector, encoded_embeddings[:-1]]
        )  # length N + 1
        conditioning_vectors = self.decoder(inputs=conditioning_vectors)

        # decode the length and items independently
        conditioning_len = conditioning_vectors[0]
        pred_len = self.len_codec.decode(
            conditioning_vector=conditioning_len,
            context=None,
            shared_codecs=shared_codecs,
        )

        conditioning_items = conditioning_vectors[1:]

        pred_items = shared_codecs.decode(
            self.subcodec_in, conditioning_items, subcontexts, vmapped=True
        )

        return pred_len, pred_items

    def sample(
        self, conditioning_vector: Embedding, shared_codecs: SharedCodecs
    ) -> Tuple[ListObservation, Embedding]:
        # sample the length
        sampled_len, embedding_len = self.len_codec.sample(
            conditioning_vector, shared_codecs
        )

        # sample the items auto-regressively
        def sample_one(self, carry):
            embeddings, i = carry

            encoded_embeddings = self.encoder(embeddings)
            conditioned_encoded_embeddings = jnp.vstack(
                [conditioning_vector, encoded_embeddings[:-1]]
            )  # length N + 1

            conditioning_vectors = self.decoder(inputs=conditioned_encoded_embeddings)

            conditioning_vector_i = conditioning_vectors[i + 1]
            rng = self.make_rng("sample")
            sample_i, embedding_i = shared_codecs.sample(
                self.subcodec_in, conditioning_vector_i, rng
            )
            embeddings = embeddings.at[i + 1].set(embedding_i)

            carry = embeddings, i + 1
            y = sample_i
            return carry, y

        embeddings = jnp.zeros(shape=(self.max_len, self.embed_dim))
        embeddings = jnp.vstack([embedding_len, embeddings])  # length N + 1

        (embeddings, _), samples = nn.scan(
            target=sample_one,
            variable_broadcast="params",
            split_rngs={"sample": True},
            length=self.max_len,
        )(self, (embeddings, 0))

        encoded_embeddings = self.encoder(embeddings)
        return (sampled_len, samples), encoded_embeddings[-1]

    def loss(
        self,
        x: ListObservation,
        prediction: ListPrediction,
        shared_codecs: SharedCodecs,
    ) -> jnp.ndarray:
        x_len, x_items = x
        pred_len, pred_items = prediction

        loss_len = self.len_codec.loss(
            x=x_len, prediction=pred_len, shared_codecs=shared_codecs
        )

        mask = jnp.arange(self.buffer_size) < x_len

        losses_item = (
            shared_codecs.loss(self.subcodec_in, x_items, pred_items, vmapped=True)
            * mask
        ).mean()  # / x_len
        # TODO what loss do we want for a list? NLL is too restrictive (and badly conditioned for long lists)
        return {"loss_len": loss_len, "avg_loss_items": losses_item}

    def example(self, shared_codecs: SharedCodecs):
        example_len = jnp.array(self.max_len - 1)

        # stack by hand instead of using the vmapped subcodec because it only exists after `setup` is done
        example_item = shared_codecs.example(self.subcodec_in)
        example_items_list = [example_item for _ in range(self.buffer_size)]
        example_items = jax.tree_map(lambda *s: jnp.stack(s), *example_items_list)

        return example_len, example_items
