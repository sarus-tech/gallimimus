from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp

Observation = Any
Context = Any
Prediction = Any

Embedding = jax.Array


@dataclass
class SharedCodecs:
    """TODO."""

    shared_models_dict: flax.core.FrozenDict
    shared_params_dict: flax.core.FrozenDict

    def _get_shared_module_fn(self, model_name, method_name, vmapped) -> Any:
        shared_model = self.shared_models_dict[model_name]
        shared_params = self.shared_params_dict[model_name]

        def output_fn(*args, **kwargs):
            return shared_model.apply(
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

    def update_params(self, model_name, params):
        model_dict = self.shared_models_dict
        params_dict = self.shared_params_dict.copy(add_or_replace={model_name: params})
        return SharedCodecs(model_dict, params_dict)

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


@dataclass
class MockSharedCodecs:
    shared_models_dict: flax.core.FrozenDict
    shared_params_dict: flax.core.FrozenDict
    embed_dim: int

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
        rng,
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

    def update_params(self, model_name, params):
        return self
