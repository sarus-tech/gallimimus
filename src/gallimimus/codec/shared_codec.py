from __future__ import annotations

from dataclasses import dataclass

import jax

import jax.numpy as jnp

from typing import Tuple, Any


Observation = Any  # of shape () and dtype int
Context = Any
Prediction = Any  # un-normalized logits of shape (vocab_size,)

Embedding = jax.Array


@dataclass
class SharedCodecs:
    """TODO"""

    shared_models_dict: dict
    shared_params_dict: dict

    def _get_shared_module_fn(self, model_name, method_name, vmapped) -> Any:
        shared_model = self.shared_models_dict[model_name]
        shared_params = self.shared_params_dict[model_name]

        output_fn = lambda *args, **kwargs: shared_model.apply(
            {"params": shared_params},
            *args,
            **kwargs,
            shared_codecs=self,
            method=method_name,
        )

        if vmapped:
            output_fn = jax.vmap(
                output_fn,
                in_axes=0,
            )

        return output_fn

    def encode(
        self, model_name, x: Observation, vmapped=False
    ) -> Tuple[Embedding, Context]:
        encoding_fn = self._get_shared_module_fn(
            model_name=model_name, method_name="encode", vmapped=vmapped
        )

        return encoding_fn(x=x)

    def decode(
        self,
        model_name,
        conditioning_vector: Embedding,
        context: Context,
        vmapped=False,
    ) -> Prediction:
        decoding_fn = self._get_shared_module_fn(
            model_name=model_name,
            method_name="decode",
            vmapped=vmapped,
        )

        return decoding_fn(
            conditioning_vector=conditioning_vector,
            context=context,
        )

    def sample(
        self,
        model_name,
        conditioning_vector: Embedding,
        rng,
        vmapped=False,
    ) -> Tuple[Observation, Embedding]:
        sampling_fn = self._get_shared_module_fn(
            model_name=model_name,
            method_name="sample",
            vmapped=vmapped,
        )

        return sampling_fn(
            conditioning_vector=conditioning_vector,
            rngs={"sample": rng},
        )

    def loss(
        self,
        model_name,
        x: Observation,
        prediction: Prediction,
        vmapped=False,
    ) -> jnp.ndarray:
        loss_fn = self._get_shared_module_fn(
            model_name=model_name,
            method_name="loss",
            vmapped=vmapped,
        )

        return loss_fn(
            x=x,
            prediction=prediction,
        )

    def example(self, model_name):
        return self.shared_models_dict[model_name].example(shared_codecs=self)


class MockSharedCodecs:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def encode(
        self,
        model_name,
        x: Observation,
        vmapped=False,
    ) -> Tuple[Embedding, Context]:
        embedding = jnp.zeros((self.embed_dim,))
        context = None
        return embedding, context

    def decode(
        self,
        model_name,
        conditioning_vector: Embedding,
        context: Context,
        vmapped=False,
    ) -> Prediction:
        return None

    def sample(
        self,
        model_name,
        conditioning_vector: Embedding,
        vmapped=False,
    ) -> Tuple[Observation, Embedding]:
        sample = None
        embedding = jnp.zeros((self.embed_dim,))
        return sample, embedding

    def loss(
        self,
        model_name,
        x: Observation,
        prediction: Prediction,
        vmapped=False,
    ) -> jnp.ndarray:
        return jnp.array(0.0)

    def example(self, model_name):
        return None
