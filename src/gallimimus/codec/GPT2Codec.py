from __future__ import annotations

import jax

import jax.numpy as jnp
import flax.linen as nn

from typing import Tuple

from gallimimus.codec.abstract_codec import Codec, Embedding

CategoricalObservation = jax.Array  # of shape () and dtype int
CategoricalContext = None
CategoricalPrediction = jax.Array  # un-normalized logits of shape (vocab_size,)


from transformers import AutoTokenizer, FlaxGPT2Model

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = FlaxGPT2Model.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

outputs = model(**inputs)


class SharedCodec(Codec):
    """TODO

    An observation is TODO

    model = FlaxGPT2Model.from_pretrained("gpt2")
    apply_fn = lambda params, input, **kwargs: model(
        **input, params=params["params"], **kwargs
    )
    model_dict[encode_name]
    :param embed_dim: size of the embeddings."""

    model_encode_name: str
    model_decode_name: str

    def setup(self):
        pass

    def encode(
        self, x: CategoricalObservation, model_dict
    ) -> Tuple[Embedding, CategoricalContext]:
        assert x.shape == ()
        model_apply, params = model_dict[self.model_encode_name]

        outputs = model_apply(params, x)

        last_hidden_state = outputs.last_hidden_state

        embedding = last_hidden_state[-1]
        return embedding, None

    def decode(
        self, conditioning_vector: Embedding, context: CategoricalContext, model_dict
    ) -> CategoricalPrediction:
        model_apply, params = model_dict[self.model_decode_name]

        prediction = model_apply(params, conditioning_vector)
        return prediction

    def sample(
        self, conditioning_vector: Embedding, model_dict
    ) -> Tuple[CategoricalObservation, Embedding]:
        assert conditioning_vector.shape == (self.embed_dim,)
        prediction = self.decode(
            conditioning_vector=conditioning_vector, context=None, model_dict={}
        )
        rng = self.make_rng(name="sample")
        sample = jax.random.categorical(
            key=rng,
            logits=prediction,
        )

        embedding, _ = self.encode(x=sample, model_dict={})
        return sample, embedding

    def loss(
        self, x: CategoricalObservation, prediction: CategoricalPrediction, model_dict
    ) -> jnp.ndarray:
        logits_normalized = jax.nn.log_softmax(x=prediction)
        loss_x = -(
            logits_normalized * jax.nn.one_hot(x=x, num_classes=self.vocab_size)
        ).sum()
        return loss_x

    def example(self):
        return jnp.array(self.vocab_size - 1)
