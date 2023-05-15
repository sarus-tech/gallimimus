import jax
import jax.numpy as jnp

from minimal_synthetic_data.model import (
    Embedding,
    Observation,
    Context,
    Prediction,
    Codec,
)
from minimal_synthetic_data.transformer import Transformer

import typing as t


# for a Struct with N columns
Observation = t.List[Observation]  # of length N
Context = t.Tuple[jax.Array, t.List[Context]]
# encoded embeddings of shape (N, embed_dim), list of subcontexts of length N
Prediction = t.List[Prediction]  # of length N


class StructCodec(Codec):
    subcodecs_in: t.List[Codec]

    n_heads: int = 4
    n_blocks: int = 1

    def setup(self):
        self.subcodecs = [subcodec.clone() for subcodec in self.subcodecs_in]
        self.encoder = Transformer(num_heads=self.n_heads, num_blocks=self.n_blocks)
        self.decoder = Transformer(num_heads=self.n_heads, num_blocks=self.n_blocks)

    def encode(self, x: Observation) -> t.Tuple[Embedding, Context]:
        # apply sub-codec encoders to each column independently
        embeddings, subcontexts = zip(
            *[
                subcodec_i.encode(x_i)
                for x_i, subcodec_i in zip(x, self.subcodecs)
            ]
        )

        # apply encoder transformer to the embeddings
        embeddings = jnp.vstack(embeddings)  # length N
        encoded_embeddings = self.encoder(inputs=embeddings)

        return encoded_embeddings[-1], (encoded_embeddings, subcontexts)

    def decode(self, conditioning_vector: Embedding, context: Context) -> Prediction:
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

    def sample(self, conditioning_vector: Embedding) -> t.Tuple[Observation, Embedding]:
        samples = []
        embeddings = jnp.zeros(
            shape=(len(self.subcodecs), self.embed_dim)
        )  # length N

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

    def loss(self, x: Observation, prediction: Prediction) -> float:
        losses = [
            subcodec.loss(x=x_i, prediction=pred_i)
            for subcodec, x_i, pred_i in zip(self.subcodecs, x, prediction)
        ]
        return jnp.array(losses).sum()

    def example(self) -> Observation:
        # iterate over self.subcodecs_in instead of self.subcodecs because they only exist after `setup` is done
        sub_examples = [subcodec.example() for subcodec in self.subcodecs_in]
        return tuple(sub_examples)
