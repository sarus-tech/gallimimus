from __future__ import annotations

import jax
import flax
import jax.numpy as jnp
import flax.linen as nn

from typing import Tuple, Callable, List

from gallimimus.codec.abstract_codec import Codec, Embedding

CategoricalObservation = jax.Array  # of shape () and dtype int
CategoricalContext = None
CategoricalPrediction = jax.Array  # un-normalized logits of shape (vocab_size,)


from lora_flax.lora import init_lora_params, lora_combine_params, FilterFunction


class LoraCodec(Codec):
    """TODO

    :param embed_dim: size of the embeddings."""

    subcodec_in: Codec
    lora_module_name: str
    filter_fn: FilterFunction
    r: int

    def setup(self):
        self.subcodec = self.subcodec_in.clone()

    @nn.compact
    def apply_lora(self, shared_dicts):
        model_dict, params_dict = shared_dicts

        pretrained_params = params_dict[self.lora_module_name]
        lora_params = self.param(
            "lora_params", init_lora_params, pretrained_params, self.filter_fn, self.r
        )

        summed_params = lora_combine_params(
            pretrained_params=pretrained_params,
            lora_params=lora_params,
        )

        updated_params_dict = params_dict.copy({self.lora_module_name: summed_params})
        return model_dict, updated_params_dict

    def encode(
        self, x: CategoricalObservation, shared_dicts
    ) -> Tuple[Embedding, CategoricalContext]:
        lora_shared_dicts = self.apply_lora(shared_dicts)
        return self.subcodec.encode(x=x, shared_dicts=lora_shared_dicts)

    def decode(
        self, conditioning_vector: Embedding, context: CategoricalContext, shared_dicts
    ) -> CategoricalPrediction:
        lora_shared_dicts = self.apply_lora(shared_dicts)
        return self.subcodec.decode(
            conditioning_vector=conditioning_vector,
            context=context,
            shared_dicts=lora_shared_dicts,
        )

    def sample(
        self, conditioning_vector: Embedding, shared_dicts
    ) -> Tuple[CategoricalObservation, Embedding]:
        lora_shared_dicts = self.apply_lora(shared_dicts)
        return self.subcodec.sample(
            conditioning_vector=conditioning_vector, shared_dicts=lora_shared_dicts
        )

    def loss(
        self, x: CategoricalObservation, prediction: CategoricalPrediction, shared_dicts
    ) -> jnp.ndarray:
        lora_shared_dicts = self.apply_lora(shared_dicts)
        return self.subcodec.loss(
            x=x, prediction=prediction, shared_dicts=lora_shared_dicts
        )

    def example(self, shared_dicts):
        return self.subcodec_in.example(shared_dicts=shared_dicts)
