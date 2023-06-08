from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from lora_flax import FilterFunction
from lora_flax.lora import init_lora_params, lora_combine_params

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
        params_dict = shared_codecs.shared_params_dict

        if self.lora_module_name not in params_dict:
            raise KeyError(
                f"Pretrained parameters for `{self.lora_module_name}` were not provided. "
            )
        pretrained_params = params_dict[self.lora_module_name]

        lora_params = self.param(
            "lora_params", init_lora_params, pretrained_params, self.filter_fn, self.r
        )

        summed_params = lora_combine_params(
            pretrained_params=pretrained_params,
            lora_params=lora_params,
        )

        return shared_codecs.update_params(
            model_name=self.lora_module_name,
            params=summed_params,
        )

    def encode(
        self, x: CategoricalObservation, shared_codecs: SharedCodecs
    ) -> Tuple[Embedding, CategoricalContext]:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        return lora_shared_codecs.encode(model_name=self.subcodec_in, x=x)

    def decode(
        self, conditioning_vector: Embedding, context: CategoricalContext, shared_codecs
    ) -> CategoricalPrediction:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        return lora_shared_codecs.decode(
            model_name=self.subcodec_in,
            conditioning_vector=conditioning_vector,
            context=context,
        )

    def sample(
        self, conditioning_vector: Embedding, shared_codecs: SharedCodecs
    ) -> Tuple[CategoricalObservation, Embedding]:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        rng = self.make_rng(name="sample")

        return lora_shared_codecs.sample(
            model_name=self.subcodec_in,
            conditioning_vector=conditioning_vector,
            rng=rng,
        )

    def loss(
        self,
        x: CategoricalObservation,
        prediction: CategoricalPrediction,
        shared_codecs: SharedCodecs,
    ) -> jnp.ndarray:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        return lora_shared_codecs.loss(
            model_name=self.subcodec_in, x=x, prediction=prediction
        )

    def example(self, shared_codecs: SharedCodecs):
        return shared_codecs.example(self.subcodec_in)
