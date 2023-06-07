from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from lora_flax import LoRA, FilterFunction

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

        if self.lora_module_name not in params_dict:
            raise KeyError(
                f"Pretrained parameters for `{self.lora_module_name}` were not provided. "
            )
        pretrained_params = params_dict[self.lora_module_name]

        lora_codec = LoRA(
            target_module=model_dict[self.lora_module_name],
            pretrained_params=pretrained_params,
            filter_fn=self.filter_fn,
            r=self.r,
            methods={
                "encode": [],
                "decode": [],
                "loss": [],
                "sample": ["sample"],
                "init_pass": [],
            },
        )

        def init_fn(rngs):
            return lora_codec.init(rngs, model_dict, params_dict, method="init_pass")[
                "params"
            ]

        lora_params = self.param(
            "lora_params",
            init_fn,
        )

        return shared_codecs.update(
            model_name=self.lora_module_name,
            model=lora_codec,
            params=lora_params,
        )

    def encode(
        self, x: CategoricalObservation, shared_codecs: SharedCodecs
    ) -> Tuple[Embedding, CategoricalContext]:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        return lora_shared_codecs.encode(model_name=self.lora_module_name, x=x)

    def decode(
        self, conditioning_vector: Embedding, context: CategoricalContext, shared_codecs
    ) -> CategoricalPrediction:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        return lora_shared_codecs.decode(
            model_name=self.lora_module_name,
            conditioning_vector=conditioning_vector,
            context=context,
        )

    def sample(
        self, conditioning_vector: Embedding, shared_codecs: SharedCodecs
    ) -> Tuple[CategoricalObservation, Embedding]:
        lora_shared_codecs = self.apply_lora(shared_codecs)
        rng = self.make_rng(name="sample")

        return lora_shared_codecs.sample(
            model_name=self.lora_module_name,
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
            model_name=self.lora_module_name, x=x, prediction=prediction
        )

    def example(self, shared_codecs: SharedCodecs):
        return shared_codecs.example(self.subcodec_in)
