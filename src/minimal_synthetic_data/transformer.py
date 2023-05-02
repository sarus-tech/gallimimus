"""Transformer-based language models.

from https://github.com/google/flax/blob/main/examples/nlp_seq/models.py"""

import jax.numpy as jnp

import flax.linen as nn


class Encoder1DBlock(nn.Module):
    """Causal Encoder Block."""

    num_heads: int
    mlp_dim: int = 0

    @nn.compact
    def __call__(self, inputs):
        mask = jnp.tri(N=len(inputs))
        x = nn.SelfAttention(num_heads=self.num_heads)(inputs_q=inputs, mask=mask)
        x = x + inputs
        return x


class Transformer(nn.Module):
    """Causal Transformer Model."""

    num_heads: int
    num_blocks: int

    @nn.compact
    def __call__(self, inputs):
        """Applies Transformer model on the inputs."""
        # assert inputs.ndim == 2  # (len, embed_dim)
        x = inputs
        for _ in range(self.num_blocks):
            x = Encoder1DBlock(num_heads=self.num_heads)(x)
        return x
