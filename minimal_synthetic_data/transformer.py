"""Transformer-based language models.

from https://github.com/google/flax/blob/main/examples/nlp_seq/models.py"""

import jax.numpy as jnp

import flax.linen as nn


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.
    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    num_heads: int
    mlp_dim: int = 0

    @nn.compact
    def __call__(self, inputs):
        """Applies Encoder1DBlock module."""
        mask = jnp.tri(N=len(inputs))

        # Attention block.
        # x = nn.LayerNorm()(inputs)
        x = nn.SelfAttention(num_heads=self.num_heads)(inputs_q=inputs, mask=mask)

        # x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # # MLP block.
        # y = nn.LayerNorm()(x)
        # y = nn.Dense(self.mlp_dim)(y)
        # y = nn.elu(x)
        # return x + y
        return x


class Transformer(nn.Module):
    """Transformer Model for sequence tagging."""

    num_heads: int
    num_blocks: int

    @nn.compact
    def __call__(self, inputs):
        """Applies Transformer model on the inputs.
        Args:
          inputs: input data
        Returns:
          output of a transformer encoder.
        """
        assert inputs.ndim == 2  # (len, embed_dim)

        # x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=not train)
        # x = AddPositionEmbs(config)(x)
        x = inputs
        for _ in range(self.num_blocks):
            x = Encoder1DBlock(num_heads=self.num_heads)(x)

        # x = nn.LayerNorm()(x)
        return x
