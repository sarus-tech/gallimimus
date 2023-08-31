from typing import Any, Tuple

import jax
from jax import numpy as jnp

Array = Any
EPS = jnp.power(10.0, -5)


def interpolate(x: float, quantiles: Array, embeddings: Array) -> Array:
    idx = jnp.searchsorted(quantiles, x) - 1
    embedding = embeddings[:-1][idx]
    return embedding


def make_bucket_probs(quantiles: Array, logits: Array) -> Array:
    len_quantiles = quantiles[1:] - quantiles[:-1]

    l_q = logits[1:-2] * len_quantiles
    l_l = jnp.array([logits[0]])
    l_r = jnp.array([logits[-1]])

    return jnp.concatenate([l_l, l_q, l_r])
    """if we want the logits to represent the proba of the bucket:"""
    # l_r = jnp.array([logits[-1]])
    # return jnp.concatenate([logits[:-2], l_r])


def sample_regular(
    rng: jax.random.KeyArray,
    sample_idx: int,
    quantiles: Array,
    logits: Array,
    log_mus: Tuple[float, float],
) -> float:
    """Sample a regular bucket."""

    sample_idx_q = sample_idx - 1
    q_a, q_b = quantiles[sample_idx_q], quantiles[sample_idx_q + 1]

    sample = jax.random.uniform(rng, minval=q_a, maxval=q_b)
    return sample


def log_likelihood_regular(
    x: float,
    quantiles: Array,
    logits_full: Array,
    log_mus: Tuple[float, float],
) -> float:
    """Likelyhood in a regular bucket (not a tail one)"""

    n = len(quantiles) - 1
    assert len(logits_full) == n + 3

    idx = 1 + jnp.searchsorted(quantiles[1:-1], x)

    q_a, q_b = quantiles[idx - 1], quantiles[idx]
    bucket_len = q_b - q_a

    bucket_len = jnp.maximum(bucket_len, EPS)

    x_log_prob = jnp.log(1 / bucket_len)
    """If we want the logits to represent the proba of the bucket."""
    # x_log_prob = 0

    return x_log_prob
