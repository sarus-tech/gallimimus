from __future__ import annotations

from typing import Tuple, Any, Dict

import flax
import flax.linen as nn
import jax.numpy as jnp
import jax.random
import optax
from jax import lax
from transformers import AutoTokenizer

from gallimimus.codec.abstract_codec import Codec, Embedding
from gallimimus.shared_codecs import SharedCodecs

GPTObservation = Dict  # of shape TODO and dtype TODO
GPTContext = Any  # TODO
GPTPrediction = jax.Array  # un-normalized logits of shape (TODO, N_tokens,)


class TextCodec(Codec):
    """Codec for categorical data"""

    n_tokens: int
    """number of tokens modeled by the context"""
    max_length: int

    model_name: str

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        super().__post_init__()

    def setup(self):
        # config = AutoConfig.from_pretrained("gpt2")
        #
        # model = FlaxGPT2LMHeadModel.from_config(config)
        #
        # self.gpt2_model = FlaxAutoModel.from_pretrained(self.model_name)

        self.bos_token_id = self.tokenizer.bos_token_id

        self.context_projection = nn.Dense(
            self.n_tokens * 768
        )  # embedding dim of distilled english gpt2
        self.embedding_proj = nn.Dense(self.embed_dim)
        self.attention_embedding = self.param(
            "query_embedding", nn.initializers.uniform(), (1, 768)
        )
        self.cross_attention = nn.MultiHeadDotProductAttention(4)

    def encode(
        self, x: GPTObservation, shared_codecs: SharedCodecs
    ) -> Tuple[Embedding, GPTContext]:
        inputs = x
        gpt2_model = shared_codecs.shared_models_dict[self.model_name]
        gpt2_params = shared_codecs.shared_params_dict[self.model_name]

        # need to expand dims
        input_ids = inputs["input_ids"][None, :]
        position_ids = inputs["position_ids"][None, :]
        attention_mask = inputs["attention_mask"][None, :]

        # compute summarization
        output = gpt2_model.apply(
            variables={"params": gpt2_params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        last_hidden_state = output.last_hidden_state[0]
        vector = self.cross_attention(
            inputs_q=self.attention_embedding,
            inputs_kv=last_hidden_state,
            mask=attention_mask,
        )[0]
        embedding = self.embedding_proj(vector)

        # shift position_ids by n_tokens
        position_ids += jnp.full_like(position_ids, self.n_tokens)
        # compute embeddings for intermediate contexts

        pos_emb = gpt2_model.wpe.apply(
            variables={"params": gpt2_params["wpe"]}, inputs=position_ids.astype("i4")
        )

        tok_emb = gpt2_model.wte.apply(
            variables={"params": gpt2_params["wte"]}, inputs=input_ids.astype("i4")
        )

        intermediate_embeddings = pos_emb + tok_emb

        # consider that end token is provided already
        # in the attention mask
        intermediate = {
            "embeddings": intermediate_embeddings,
            "attention_mask": attention_mask,
        }

        return embedding, intermediate

    def decode(
        self,
        conditioning_vector: Embedding,
        context: GPTContext,
        shared_codecs: SharedCodecs,
    ) -> GPTPrediction:
        gpt2_model = shared_codecs.shared_models_dict[self.model_name]
        gpt2_params = shared_codecs.shared_params_dict[self.model_name]

        start_tokens = self.context_projection(conditioning_vector)
        start_tokens = jnp.reshape(start_tokens, (self.n_tokens, 768))
        hidden_states = jnp.concatenate(
            [start_tokens[None, :], context["embeddings"]], axis=1
        )

        # update attention_mask with context tokens
        attention_mask = jnp.concatenate(
            [
                jnp.ones((1, self.n_tokens)),
                context["attention_mask"],
            ],
            axis=-1,
        )
        # compute transformer_blocks+layer_norm+head to get logits
        hidden_states = gpt2_model.h.apply(
            {"params": gpt2_params["h"]}, hidden_states, attention_mask
        )[0]

        hidden_states = gpt2_model.ln_f.apply(
            {"params": gpt2_params["ln_f"]}, hidden_states
        )

        predictions = gpt2_model.wte.apply(
            {"params": gpt2_params["wte"]}, hidden_states, method="attend"
        )

        return predictions[0, self.n_tokens - 1 : -1, :]

    def sample(
        self, conditioning_vector: Embedding, shared_codecs: SharedCodecs
    ) -> Tuple[GPTObservation, Embedding]:
        gpt2_model = shared_codecs.shared_models_dict[self.model_name]
        gpt2_params = shared_codecs.shared_params_dict[self.model_name]

        def _sample_gpt2(self, carry: SampleState, sequences: jnp.ndarray):
            """"""

            curr_len = carry.cur_len
            gpt2_model = carry.model
            embeddings = carry.running_embeddings
            attention_mask = carry.attention_mask[None, :]
            switch_to_zero_attention = carry.switch_to_zero_attention

            # Compute logits
            hidden_states = gpt2_model.h.apply(
                {"params": gpt2_params["h"]}, embeddings[None, :], attention_mask
            )[0]

            hidden_states = gpt2_model.ln_f.apply(
                {"params": gpt2_params["ln_f"]}, hidden_states
            )

            predictions = gpt2_model.wte.apply(
                {"params": gpt2_params["wte"]}, hidden_states, method="attend"
            )

            logits = predictions[0].at[curr_len - 1].get()
            # Sample token
            next_token = jax.random.categorical(self.make_rng("sample"), logits)
            # next_token = jnp.argmax(logits)
            next_sequences = lax.dynamic_update_slice(sequences, next_token, ())
            # Update embeddings
            position_id_next_token = jnp.array(curr_len)
            pos_emb = gpt2_model.wpe.apply(
                {"params": gpt2_params["wpe"]}, position_id_next_token
            )
            tok_emb = gpt2_model.wte.apply({"params": gpt2_params["wte"]}, next_token)
            embedding_next_token = pos_emb + tok_emb

            embeddings = embeddings.at[curr_len].set(embedding_next_token)
            updated_attention_mask = carry.attention_mask.at[curr_len].set(
                switch_to_zero_attention
            )
            return (
                SampleState(
                    cur_len=curr_len + 1,
                    running_embeddings=embeddings,
                    model=gpt2_model,
                    attention_mask=updated_attention_mask,
                    switch_to_zero_attention=switch_to_zero_attention
                    * jnp.not_equal(next_token, self.bos_token_id),
                ),
                next_sequences,
            )

        _sample_gpt2 = nn.scan(
            target=_sample_gpt2,
            variable_broadcast="params",
            split_rngs={"sample": True},
        )

        start_embeddings = jnp.zeros(shape=(self.max_length, 768))
        bias_embeddings = self.context_projection(conditioning_vector)
        bias_embeddings = jnp.reshape(bias_embeddings, (self.n_tokens, 768))
        start_embeddings = jnp.concatenate([bias_embeddings, start_embeddings], axis=-2)
        pad_token_id = jnp.array(self.bos_token_id + 1, dtype=jnp.int32)
        init_sequence = jnp.full((self.max_length,), pad_token_id, dtype=jnp.int32)

        attention_mask = jnp.concatenate(
            [
                jnp.ones((self.n_tokens,)),
                jnp.zeros(
                    self.max_length,
                ),
            ],
            axis=0,
        )
        # initialize state
        init_carry = SampleState(
            cur_len=self.n_tokens,
            running_embeddings=start_embeddings,
            model=gpt2_model,
            attention_mask=attention_mask,
            switch_to_zero_attention=1,
        )
        carry, samples = self._sample_gpt2(init_carry, init_sequence)

        # recompute for last embedding
        hidden_states = gpt2_model.h.apply(
            {"params": gpt2_params["h"]},
            carry.running_embeddings[None, :],
            carry.attention_mask,
        )[0]
        hidden_states = gpt2_model.ln_f.apply(
            {"params": gpt2_params["ln_f"]}, hidden_states
        )[0]

        vector = self.cross_attention(
            inputs_q=self.attention_embedding,
            inputs_kv=hidden_states,
            mask=carry.attention_mask,
        )[0]
        return samples, self.embedding_proj(vector)

    def loss(
        self,
        x: GPTObservation,
        prediction: GPTPrediction,
        shared_codecs: SharedCodecs,
    ) -> jnp.ndarray:
        inputs = x

        loss = optax.softmax_cross_entropy_with_integer_labels(
            labels=inputs["input_ids"].astype("i4"), logits=prediction
        )
        loss *= inputs["attention_mask"]
        loss = jnp.sum(loss, axis=-1) / jnp.sum(inputs["attention_mask"], axis=-1)
        return loss

    def example(self, shared_codecs: SharedCodecs) -> GPTObservation:
        tokens = self.tokenizer("example", padding="longest")
        tokens = {k: jnp.array(v) for k, v in tokens.items()}
        tokens["position_ids"] = jnp.arange(len(tokens["input_ids"]))
        return tokens


@flax.struct.dataclass
class SampleState:
    """State used to process sampling"""

    cur_len: int
    running_embeddings: jnp.ndarray
    attention_mask: jnp.ndarray
    model: Any = flax.struct.field(pytree_node=False)
    switch_to_zero_attention: int
