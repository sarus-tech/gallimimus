"""Transformer-based language models."""
import flax.linen as nn
import jax.numpy as jnp
import jax.tree_util
from transformers import GPT2Config
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2BlockCollection


class Transformer(nn.Module):
    """Causal Transformer Model."""

    num_heads: int
    num_blocks: int
    embed_dim: int

    def setup(self) -> None:
        config = GPT2Config(
            n_head=self.num_heads,
            n_layer=self.num_blocks,
            n_embd=self.embed_dim,
        )

        self.transformer = FlaxGPT2BlockCollection(config=config)

    @nn.compact
    def __call__(self, inputs):
        """Applies the GPT2 transformer model on the inputs."""
        # assert inputs.ndim == 2  # (len, embed_dim)

        inputs = jax.tree_util.tree_map(
            lambda arr: jnp.expand_dims(arr, axis=0), inputs
        )
        outputs = self.transformer(
            hidden_states=inputs,
        )[0]

        outputs = jax.tree_util.tree_map(lambda arr: jnp.squeeze(arr, axis=0), outputs)
        return outputs
