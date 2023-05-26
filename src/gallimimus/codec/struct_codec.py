from __future__ import annotations

import jax
import jax.numpy as jnp

from gallimimus.codec.abstract_codec import (
    Codec,
    Embedding,
    Observation,
    Context,
    Prediction,
)
from gallimimus.transformer import Transformer
from typing import List, Tuple


# for a Struct with N columns
StructObservation = List[Observation]  # of length N
StructContext = Tuple[jax.Array, List[Context]]
# encoded embeddings of shape (N, embed_dim), list of subcontexts of length N
StructPrediction = List[Prediction]  # of length N


class StructCodec(Codec):
    """A codec for tabular data where the columns are generated each using their own sub-codec, as provided
    by ``subcodecs_in``.

    If ``subcodecs_in = [subcodec_1, ..., subcodec_n]`` and ``subcodec_i`` handles type SubObservation_i,
    an observation has type ``Tuple[SubObservation_1, ..., SubObservation_n]``.

    :param embed_dim: Size of the embeddings.
    :param subcodecs_in: List of the Codecs used for each column.
    :param n_heads: Number of transformer heads.
    :param n_blocks: Number of transformer blocks.
    """

    subcodecs_in: List[Codec]
    n_heads: int = 4
    n_blocks: int = 1

    def setup(self):
        self.subcodecs = [subcodec.clone() for subcodec in self.subcodecs_in]
        self.encoder = Transformer(num_heads=self.n_heads, num_blocks=self.n_blocks)
        self.decoder = Transformer(num_heads=self.n_heads, num_blocks=self.n_blocks)

    def encode(self, x: StructObservation) -> Tuple[Embedding, StructContext]:
        # apply sub-codec encoders to each column independently
        embeddings, subcontexts = zip(
            *[subcodec_i.encode(x_i) for x_i, subcodec_i in zip(x, self.subcodecs)]
        )

        # apply encoder transformer to the embeddings
        embeddings = jnp.vstack(embeddings)  # length N
        encoded_embeddings = self.encoder(inputs=embeddings)

        return encoded_embeddings[-1], (encoded_embeddings, subcontexts)

    def decode(
        self, conditioning_vector: Embedding, context: StructContext
    ) -> StructPrediction:
        encoded_embeddings, subcontexts = context

        # apply decoder transformer to the *conditioned* embeddings
        conditioning_contexts = jnp.vstack(
            [conditioning_vector, encoded_embeddings[:-1]]
        )  # length N
        conditioning_contexts = self.decoder(inputs=conditioning_contexts)

        # apply subcodec decoders to each column independently
        sub_predictions = [
            subcodec_i.decode(conditioning_context_i, subcontext_i)
            for conditioning_context_i, subcontext_i, subcodec_i in zip(
                conditioning_contexts,
                subcontexts,
                self.subcodecs,
            )
        ]

        return sub_predictions

    def sample(
        self, conditioning_vector: Embedding
    ) -> Tuple[StructObservation, Embedding]:
        samples = []
        embeddings = jnp.zeros(shape=(len(self.subcodecs), self.embed_dim))  # length N

        for i, subcodec_i in enumerate(self.subcodecs):
            # encode what was previously sampled
            encoded_embeddings = self.encoder(embeddings)

            # decode the *conditioned* embeddings
            conditioned_encoded_embeddings = jnp.vstack(
                [conditioning_vector, encoded_embeddings[:-1]]
            )  # length N
            conditioning_vector_i = self.decoder(inputs=conditioned_encoded_embeddings)[
                i
            ]

            # sample from the next column
            sample_i, embedding_i = subcodec_i.sample(
                conditioning_vector=conditioning_vector_i
            )

            samples.append(sample_i)
            embeddings = embeddings.at[i].set(embedding_i)

        encoded_embeddings = self.encoder(embeddings)
        return samples, encoded_embeddings[-1]

    def loss(self, x: StructObservation, prediction: StructPrediction) -> jnp.ndarray:
        losses = [
            subcodec.loss(x=x_i, prediction=pred_i)
            for subcodec, x_i, pred_i in zip(self.subcodecs, x, prediction)
        ]
        return jnp.array(losses).sum()

    def example(self) -> StructObservation:
        # iterate over self.subcodecs_in instead of self.subcodecs because they only exist after `setup` is done
        sub_examples = [subcodec.example() for subcodec in self.subcodecs_in]
        return list(sub_examples)
