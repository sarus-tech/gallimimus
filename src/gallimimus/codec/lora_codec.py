from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from lora_flax.lora import init_lora_params, lora_combine_params, FilterFunction

from gallimimus.codec.abstract_codec import Codec, Embedding
from gallimimus.shared_codecs import SharedCodecs

CategoricalObservation = jax.Array  # of shape () and dtype int
CategoricalContext = None
CategoricalPrediction = jax.Array  # un-normalized logits of shape (vocab_size,)


class LoraCodec(Codec):
    """TODO

    :param embed_dim: size of the embeddings."""

    subcodec_in: str
    lora_module_name: str
    filter_fn: FilterFunction
    r: int

    @nn.compact
    def apply_lora(self, shared_codecs: SharedCodecs):
        model_dict, params_dict = (
            shared_codecs.shared_models_dict,
            shared_codecs.shared_params_dict,
        )

        pretrained_params = params_dict[self.lora_module_name]
        lora_params = self.param(
            "lora_params", init_lora_params, pretrained_params, self.filter_fn, self.r
        )

        summed_params = lora_combine_params(
            pretrained_params=pretrained_params,
            lora_params=lora_params,
        )

        updated_params_dict = params_dict.copy({self.lora_module_name: summed_params})
        return SharedCodecs(model_dict, updated_params_dict)

    def encode(
        self, x: CategoricalObservation, shared_codecs: SharedCodecs
    ) -> Tuple[Embedding, CategoricalContext]:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        return self.subcodec.encode(x=x, shared_codecs=lora_shared_codecs)

    def decode(
        self, conditioning_vector: Embedding, context: CategoricalContext, shared_codecs
    ) -> CategoricalPrediction:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        return self.subcodec.decode(
            conditioning_vector=conditioning_vector,
            context=context,
            shared_dicts=lora_shared_codecs,
        )

    def sample(
        self, conditioning_vector: Embedding, shared_codecs: SharedCodecs
    ) -> Tuple[CategoricalObservation, Embedding]:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        return self.subcodec.sample(
            conditioning_vector=conditioning_vector, shared_dicts=lora_shared_codecs
        )

    def loss(
        self,
        x: CategoricalObservation,
        prediction: CategoricalPrediction,
        shared_codecs: SharedCodecs,
    ) -> jnp.ndarray:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        return self.subcodec.loss(
            x=x, prediction=prediction, shared_dicts=lora_shared_codecs
        )

    def example(self, shared_codecs: SharedCodecs):
        return shared_codecs.example(self.subcodec_in)
