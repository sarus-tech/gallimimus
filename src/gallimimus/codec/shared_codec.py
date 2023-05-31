from __future__ import annotations

import jax

import jax.numpy as jnp
import flax.linen as nn

from typing import Tuple, Any

from gallimimus.codec.abstract_codec import Codec, Embedding

CategoricalObservation = jax.Array  # of shape () and dtype int
CategoricalContext = None
CategoricalPrediction = jax.Array  # un-normalized logits of shape (vocab_size,)


"""
shared_dicts = model_dict, params_dict
shared_model = model_dict[shared_module_name]

shared_model["encode"].apply(params, x) -> embedding
shared_model["decode"].apply(params, v) -> prediction
shared_model["sample"].apply(params, v, rng) -> sample
shared_model["loss"].apply(params, prediction, x) -> array of shape ()
"""


class SharedCodec(Codec):
    """TODO


    :param embed_dim: size of the embeddings."""

    shared_module_name: str

    def exec_shared_module(self, method_name, shared_dicts, *args, **kwargs) -> Any:
        model_dicts, params_dicts = shared_dicts
        shared_model = model_dicts[self.shared_module_name]

        shared_params = params_dicts[self.shared_module_name]
        output = shared_model.apply(
            {"params": shared_params},
            *args,
            **kwargs,
            shared_dicts=shared_dicts,
            method=method_name,
        )
        return output

    def encode(
        self, x: CategoricalObservation, shared_dicts
    ) -> Tuple[Embedding, CategoricalContext]:
        assert x.shape == ()

        embedding, context = self.exec_shared_module(
            x=x,
            method_name="encode",
            shared_dicts=shared_dicts,
        )

        return embedding, context

    def decode(
        self, conditioning_vector: Embedding, context: CategoricalContext, shared_dicts
    ) -> CategoricalPrediction:
        prediction = self.exec_shared_module(
            method_name="decode",
            shared_dicts=shared_dicts,
            conditioning_vector=conditioning_vector,
            context=context,
        )

        return prediction

    def sample(
        self, conditioning_vector: Embedding, shared_dicts
    ) -> Tuple[CategoricalObservation, Embedding]:
        assert conditioning_vector.shape == (self.embed_dim,)

        rng = self.make_rng(name="sample")

        sample, embedding = self.exec_shared_module(
            method_name="sample",
            shared_dicts=shared_dicts,
            conditioning_vector=conditioning_vector,
            rngs={"sample": rng},
        )

        return sample, embedding

    def loss(
        self, x: CategoricalObservation, prediction: CategoricalPrediction, shared_dicts
    ) -> jnp.ndarray:
        loss_x = self.exec_shared_module(
            method_name="loss",
            shared_dicts=shared_dicts,
            x=x,
            prediction=prediction,
        )
        return loss_x

    def example(self, shared_dicts):
        model_dicts, params_dicts = shared_dicts
        shared_model = model_dicts[self.shared_module_name]

        return shared_model.example(shared_dicts)
