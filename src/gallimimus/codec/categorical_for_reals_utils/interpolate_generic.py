from functools import partial
from typing import Any, Tuple

import jax
from jax import numpy as jnp

import gallimimus.codec.categorical_for_reals_utils.interpolate_0 as Interpolate_0
import gallimimus.codec.categorical_for_reals_utils.interpolate_1 as Interpolate_1

Array = Any
"""
Warning:
- quantiles is of len : n_quantiles + 1 (has right endpoint)
- embeddings is of len : n_quantiles + 1 (one embedding per marker)


TODO have to choose between 
1. the logits are evaluation of the distribution at the edges of the buckets
2. the logits are the probability of each bucket

For the tail buckets only 2. makes sense
For the other buckets, 1. is easier to understand

We will say that
- logits[0] and logits[-1] represent the overall probability of the respectively left 
and right tail buckets (ie option 2.)
- logits[1:-1] represent the evaluation of the distribution at the edges of the buckets 
(ie option 1.)

for the exponential decaying tails, this means we can compute the parameter lambda:
    lambda = p_edge / p_bucket
"""


interpolate_switch_list = [
    Interpolate_0.interpolate,
    Interpolate_1.interpolate,
]

make_bucket_probs_switch_list = [
    Interpolate_0.make_bucket_probs,
    Interpolate_1.make_bucket_probs,
]

sample_regular_switch_list = [
    Interpolate_0.sample_regular,
    Interpolate_1.sample_regular,
]

log_likelihood_regular_switch_list = [
    Interpolate_0.log_likelihood_regular,
    Interpolate_1.log_likelihood_regular,
]

EPS = 1e-3


@jax.jit
def find_region(quantiles, x):
    return (x >= quantiles[0] - EPS).astype(int) + (x > quantiles[-1] + EPS).astype(int)


### INTERPOLATION FUNCTIONS


@partial(jax.jit, static_argnums=(0,))
def interpolate(
    interpolate_mode: int, x: float, quantiles: Array, embeddings: Array
) -> Array:
    # return jax.lax.switch(
    #     interpolate_mode,
    #     interpolate_switch_list,
    #     x,
    #     quantiles,
    #     embeddings,
    # )
    return interpolate_switch_list[interpolate_mode](x, quantiles, embeddings)


### EMBEDDING
@partial(jax.jit, static_argnums=(0,))
def embed_full(
    interpolate_mode: int,
    inputs: float,
    quantiles: Array,
    embeddings: Array,
) -> float:
    embedding = jax.lax.switch(
        find_region(quantiles, inputs),
        [
            lambda _embeddings: _embeddings[0],
            lambda _embeddings: interpolate(
                interpolate_mode,
                inputs,
                quantiles,
                _embeddings[1:-1],
            ),
            lambda _embeddings: _embeddings[-1],
        ],
        embeddings,
    )

    return embedding


## COMPUTE THE PROBABILITY OF EACH BUCKET
@partial(jax.jit, static_argnums=(0,))
def make_bucket_probs(
    interpolate_mode: int,
    quantiles: Array,
    logits: Array,
) -> Array:
    # return jax.lax.switch(
    #     interpolate_mode,
    #     make_bucket_probs_switch_list,
    #     quantiles,
    #     logits,
    # )
    return make_bucket_probs_switch_list[interpolate_mode](quantiles, logits)


### SAMPLING FROM THE DISTRIBUTION:


def sample_left(
    interpolate_mode,
    rng: jax.random.KeyArray,
    idx: float,
    quantiles: Array,
    logits: Array,
    log_mus: Tuple[float, float],
) -> float:
    """Sample the left tail."""
    mu = jnp.exp(log_mus[0])
    delta = jax.random.exponential(rng) * mu
    return quantiles[0] - delta


def sample_right(
    interpolate_mode,
    rng: jax.random.KeyArray,
    idx: float,
    quantiles: Array,
    logits: Array,
    log_mus: Tuple[float, float],
) -> float:
    """Sample the right tail."""
    # choose a half-life for the tail
    mu = jnp.exp(log_mus[1])
    delta = jax.random.exponential(rng) * mu
    return quantiles[-1] + delta


def sample_regular(
    interpolate_mode: int,
    rng: jax.random.KeyArray,
    sample_idx: int,
    quantiles: Array,
    logits: Array,
    log_mus: Tuple[float, float],
) -> float:
    """Sample a regular bucket."""

    return jax.lax.switch(
        interpolate_mode,
        sample_regular_switch_list,
        rng,
        sample_idx,
        quantiles,
        logits,
        log_mus,
    )
    # return sample_regular_switch_list[interpolate_mode](
    #     rng,
    #     sample_idx,
    #     quantiles,
    #     logits,
    #     log_mus,
    # )


@partial(jax.jit, static_argnums=(0,))
def sample_full(
    interpolate_mode: int,
    rng: jax.random.KeyArray,
    logits_full: Array,
    quantiles: Array,
    log_mus: Tuple[float, float],
) -> float:
    rng1, rng2 = jax.random.split(rng, 2)

    log_bucket_probs = make_bucket_probs(
        interpolate_mode,
        quantiles,
        logits_full,
    )

    # TODO warning implicit softmax!
    # is it the right distrib and how does this affect log-likelihood?
    sample_idx = jax.random.categorical(rng1, log_bucket_probs, axis=-1)
    """Choose the right sampling function depending on the value of the boundaries of
    the bucket."""
    return jax.lax.switch(
        find_region(jnp.array([1, len(log_bucket_probs) - 2]), sample_idx),
        [sample_left, sample_regular, sample_right],
        interpolate_mode,
        rng2,
        sample_idx,
        quantiles,
        logits_full,
        log_mus,
    )


### COMPUTING THE likelihood


def log_likelihood_left(
    interpolate_mode,
    x: float,
    quantiles: Array,
    logits_full: Array,
    log_mus: Tuple[float, float],
    bucket_log_probs_normalized: Array,
) -> float:
    """Likelihood in the left tail."""

    n = len(quantiles) - 1
    assert len(logits_full) == n + 3

    log_mu = log_mus[0]
    log_lambd = -log_mu

    # to avoid overflow after exp
    log_lambd = jnp.minimum(log_lambd, 50.0)
    lambd = jnp.exp(log_lambd)

    delta = quantiles[0] - x

    x_log_prob = log_lambd - lambd * delta
    return bucket_log_probs_normalized[0] + x_log_prob


def log_likelihood_right(
    interpolate_mode,
    x: float,
    quantiles: Array,
    logits_full: Array,
    log_mus: Tuple[float, float],
    bucket_log_probs_normalized: Array,
) -> float:
    """Likelihood in the right tail."""

    n = len(quantiles) - 1
    assert len(logits_full) == n + 3

    # choose a half-life for the tail
    log_mu = log_mus[1]
    log_lambd = -log_mu

    # to avoid overflow after exp
    log_lambd = jnp.minimum(log_lambd, 50.0)
    lambd = jnp.exp(log_lambd)

    delta = x - quantiles[-1]
    x_log_prob = log_lambd - lambd * delta
    return bucket_log_probs_normalized[-1] + x_log_prob


def log_likelihood_regular(
    interpolate_mode: int,
    x: float,
    quantiles: Array,
    logits_full: Array,
    log_mus: Tuple[float, float],
    bucket_log_probs_normalized: Array,
) -> float:
    """Likelihood in a regular bucket."""

    idx = 1 + jnp.searchsorted(quantiles[1:-1], x)
    bucket_log_prob = bucket_log_probs_normalized[idx]

    x_log_likelihood = jax.lax.switch(
        interpolate_mode,
        log_likelihood_regular_switch_list,
        x,
        quantiles,
        logits_full,
        log_mus,
    )

    return bucket_log_prob + x_log_likelihood


@partial(jax.jit, static_argnums=(0,))
def log_likelihood(
    interpolate_mode: int,
    x: float,
    logits_full_unn: Array,
    quantiles: Array,
    log_mus: Tuple[float, float],
) -> float:
    # normalize
    bucket_log_probs = make_bucket_probs(interpolate_mode, quantiles, logits_full_unn)
    bucket_log_probs_normalized = jax.nn.log_softmax(bucket_log_probs, axis=-1)

    log_renormalization_const = jax.scipy.special.logsumexp(bucket_log_probs)
    logits_full_normalized = logits_full_unn - log_renormalization_const

    x_log_likelihood = jax.lax.switch(
        find_region(quantiles=quantiles, x=x),
        [
            log_likelihood_left,
            log_likelihood_regular,
            log_likelihood_right,
        ],
        interpolate_mode,
        x,
        quantiles,
        logits_full_normalized,
        log_mus,
        bucket_log_probs_normalized,
    )

    return x_log_likelihood
