from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any

import flax
import jax
import jax.numpy as jnp

Observation = Any
Context = Any
Prediction = Any

Embedding = jax.Array


def init_shared_trained_param_dict(rng, model_dict, pretrained_params_dict):
    trained_params_dict = {}

    for path, model in model_dict.items():
        if path not in pretrained_params_dict:
            rng, rng2 = jax.random.split(rng, 2)
            init_params = model.init(
                rngs={"params": rng},
                method=model.init_pass,
                model_dict=model_dict,
                pretrained_params_dict=pretrained_params_dict,
            )["params"]

            trained_params_dict[path] = init_params

    return flax.core.frozen_dict.freeze(trained_params_dict)


@dataclass
class SharedCodecs:
    """TODO"""

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

    def update(self, model_name, model, params):
        model_dict = self.shared_models_dict.copy(add_or_replace={model_name: model})
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

    def update(self, model_name, model, params):
        return self
