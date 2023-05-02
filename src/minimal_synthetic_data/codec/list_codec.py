import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn

from minimal_synthetic_data.model import (
    Embedding,
    Observation,
    Context,
    Prediction,
    Codec,
)
from minimal_synthetic_data.codec.categorical_codec import CategoricalCodec
from minimal_synthetic_data.transformer import Transformer

import typing as t


# For a ListCodec of buffer_size N:
Observation = t.Tuple[int, t.Any]
# real length, and stacked Pytrees of sub-observations

Context = t.Tuple[jax.Array, t.Any]
# encoded_embeddings, stacked Pytrees of sub-contexts

Prediction = t.Tuple[jnp.ndarray, t.Any]
# length logits, stacked Pytrees of sub-predictions


def vmap_clone_codec(codec_instance: Codec) -> Codec:
    """builds a *copy* of a codec instance, that is vmapped on the first axis of the leaves"""
    # code inspired by nn.Module.clone()
    codec_constructor = codec_instance.__class__
    codec_params = {
        f.name: getattr(codec_instance, f.name)
        for f in dataclasses.fields(codec_instance)
        if f.init and f.name not in ["name", "parent"]
    }

    vmapped_codec = nn.vmap(
        target=codec_constructor,
        variable_axes={"params": None},
        split_rngs={"params": False, "sample": True},
        methods=["encode", "decode", "sample", "loss"],
    )

    return vmapped_codec(**codec_params)


class ListCodec(Codec):
    """Formally the ListCodec behaves like a StructCodec[Categorical(max_len+1), subcodec, ..., subcodec] where `subcodec`
    is repeated `buffer_size` times, except that the loss of `(len_x, items_x)` only looks at the `len_x` first columns.

    For efficiency reasons, a vectorized version of the `subcodec` is used (otherwise jit compilation unrolls the loops).
    Due to this, an observation is a Pytree where the items are stacked on the first dimension of their leaves.
    """

    subcodec: Codec

    n_heads: int
    n_blocks: int

    max_len: int
    buffer_size: int

    def setup(self):
        self.len_codec = CategoricalCodec(
            embed_dim=self.embed_dim,
            vocab_size=self.max_len + 1,
        )

        self.vmapped_item_codec = vmap_clone_codec(self.subcodec)

        self.encoder = Transformer(num_heads=self.n_heads, num_blocks=self.n_blocks)
        self.decoder = Transformer(num_heads=self.n_heads, num_blocks=self.n_blocks)

    def encode(self, x: Observation) -> t.Tuple[Embedding, Context]:
        x_len, x_items = x

        # encode the length and items independently
        embedding_len, _ = self.len_codec.encode(x=x_len)
        embedding_items, subcontexts = self.vmapped_item_codec.encode(x_items)

        # transform the embeddings with the encoder
        embeddings = jnp.vstack([embedding_len, embedding_items])  # length N + 1
        encoded_embeddings = self.encoder(inputs=embeddings)

        return encoded_embeddings[-1], (encoded_embeddings, subcontexts)

    def decode(self, conditioning_vector: Embedding, context: Context) -> Prediction:
        encoded_embeddings, subcontexts = context

        # transform the *conditioned* embeddings with the decoder
        conditioning_vectors = jnp.vstack(
            [conditioning_vector, encoded_embeddings[:-1]]
        )  # length N + 1
        conditioning_vectors = self.decoder(inputs=conditioning_vectors)

        # decode the length and items independently
        conditioning_len = conditioning_vectors[0]
        pred_len = self.len_codec.decode(
            conditioning_vector=conditioning_len, context=None
        )

        conditioning_items = conditioning_vectors[1:]
        pred_items = self.vmapped_item_codec.decode(conditioning_items, subcontexts)

        return (pred_len, pred_items)

    def sample(self, conditioning_vector: Embedding) -> t.Tuple[Observation, Embedding]:
        # sample the length
        sampled_len, embedding_len = self.len_codec.sample(conditioning_vector)

        # sample the items auto-regressively
        def sample_one(self, carry):
            embeddings, i = carry

            encoded_embeddings = self.encoder(embeddings)
            conditioned_encoded_embeddings = jnp.vstack(
                [conditioning_vector, encoded_embeddings[:-1]]
            )  # length N + 1

            conditioning_vectors = self.decoder(inputs=conditioned_encoded_embeddings)
            # add a dimension because the sampling function is vectorized
            conditioning_vector_i = conditioning_vectors[i + 1][None, :]
            sample_i, embedding_i = self.vmapped_item_codec.sample(
                conditioning_vector_i
            )
            embeddings = embeddings.at[i + 1].set(embedding_i[0])

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

        # Because we used the vectorized `vmapped_item_codec.sample`, the leaves of the pytree have the shape
        # (scan_len=self.max_len, vmap_len=1, ...). This removes the extra dimension:
        samples = jax.tree_map(lambda s: s.squeeze(axis=1), samples)

        encoded_embeddings = self.encoder(embeddings)
        return (sampled_len, samples), encoded_embeddings[-1]

    def loss(self, x: Observation, prediction: Prediction) -> float:
        x_len, x_items = x
        pred_len, pred_items = prediction

        loss_len = self.len_codec.loss(x=x_len, prediction=pred_len)

        mask = jnp.arange(self.buffer_size) < x_len
        losses_item = (
            self.vmapped_item_codec.loss(x_items, pred_items) * mask
        ).sum()  # / x_len
        # TODO what loss do we want for a list? NLL is too restrictive (and badly conditioned for long lists)
        return loss_len + losses_item

    def example(self):
        example_len = jnp.array(self.max_len - 1)

        # stack by hand instead of using the vmapped subcodec because it only exists after `setup` is done
        example_items_list = [self.subcodec.example() for _ in range(self.buffer_size)]
        example_items = jax.tree_map(lambda *s: jnp.stack(s), *example_items_list)

        return (example_len, example_items)
