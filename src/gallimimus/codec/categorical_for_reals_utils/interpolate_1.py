from typing import Any, Tuple

import jax
from jax import numpy as jnp

Array = Any
EPS = 1e-5
LOG_EPS = -10000.0


def is_flat(q_a: float, q_b: float, log_p_a: float, log_p_b: float) -> bool:
    width = q_b - q_a
    almost_zero = jnp.logical_and(log_p_a < LOG_EPS, log_p_b < LOG_EPS)

    def is_flat_nonzero(width, log_p_a, log_p_b):
        log_abs_delta_p, sgn_delta_p = jax.scipy.special.logsumexp(
            jnp.array([log_p_b, log_p_a]), b=jnp.array([1.0, -1.0]), return_sign=True
        )

        return jnp.logical_or(width < EPS, log_abs_delta_p < LOG_EPS)

    is_flat = jax.lax.cond(
        almost_zero, lambda *_: True, is_flat_nonzero, width, log_p_a, log_p_b
    )
    return is_flat


def _make_log_dist_params(
    q_a: float, q_b: float, log_p_a: float, log_p_b: float
) -> Tuple[Tuple[float, float], float]:
    # do not run this function when q_a == q_b or p_a == p_b
    width = q_b - q_a
    width = jnp.maximum(width, EPS)
    log_l = jnp.log(width)

    # compute the normalization constant C = ((y1 + y2) * l) / 2
    log_C = jnp.logaddexp(log_p_a, log_p_b) + log_l - jnp.log(2)

    # normalize the edge probabilities
    log_P_a = log_p_a - log_C
    log_P_b = log_p_b - log_C

    log_abs_dP, sgn_a = jax.scipy.special.logsumexp(
        jnp.array([log_P_b, log_P_a]), b=jnp.array([1.0, -1.0]), return_sign=True
    )
    log_abs_a = log_abs_dP - log_l

    log_b = log_P_a
    return ((log_abs_a, sgn_a), log_b)


_interpolate_1 = jax.vmap(jnp.interp, in_axes=(None, None, 1))


def interpolate(x: float, quantiles: Array, embeddings: Array) -> Array:
    embeddings_2d = jnp.atleast_2d(embeddings)
    return _interpolate_1(x, quantiles, embeddings_2d)


def make_bucket_probs(quantiles: Array, logits: Array) -> Array:
    log_p_l = jnp.array([logits[0]])
    log_p_r = jnp.array([logits[-1]])

    # p(bucket i) = (p_i + p_i+1) * len_i / 2
    len_q = quantiles[1:] - quantiles[:-1]
    len_q = jnp.maximum(len_q, LOG_EPS)
    log_len_q = jnp.log(len_q)
    log_p_q = jnp.logaddexp(logits[1:-2], logits[2:-1]) + log_len_q - jnp.log(2)
    return jnp.concatenate([log_p_l, log_p_q, log_p_r])


def sample_regular(
    rng: jax.random.KeyArray,
    sample_idx: int,
    quantiles: Array,
    logits: Array,
    log_mus: Tuple[float, float],
) -> float:
    """sample a regular bucket
    we use the inverse method: sample U ~ Uniform([0,1]) and return cdf^{-1}(U)
    we sample from the distribution defined by dP(x) =
        0 if x \notin [q_a, q_b]
        1/C [ ( p_b - p_a )/(q_b - q_a) * x + p_a otherwise
    (ie the affine interpolation of the probas at the edges)
    """

    sample_idx_q = sample_idx - 1

    q_a, q_b = (
        quantiles[sample_idx_q],
        quantiles[sample_idx_q + 1],
    )
    log_p_a, log_p_b = logits[sample_idx], logits[sample_idx + 1]
    U = jax.random.uniform(rng)

    def not_flat_dist(q_a, q_b, log_p_a, log_p_b, U):
        # cdf^{-1}(U) is the solution of `a/2 * x^2 + bx = U` with
        # the following `a` and `b`:
        (log_abs_a, sgn_a), log_b = _make_log_dist_params(
            q_a=q_a, q_b=q_b, log_p_a=log_p_a, log_p_b=log_p_b
        )

        # compute Delta = b**2 + 2 * a * U
        log_Delta = jax.scipy.special.logsumexp(
            jnp.array([2 * log_b, log_abs_a]), b=jnp.array([1, sgn_a * 2 * U])
        )

        # compute sample = ( -b + jnp.sqrt( Delta ) ) / a
        log_top, sgn_top = jax.scipy.special.logsumexp(
            jnp.array([log_b, log_Delta / 2]), b=jnp.array([-1, 1]), return_sign=True
        )

        sample = sgn_a * sgn_top * jnp.exp(log_top - log_abs_a)
        return sample

    def flat_dist(q_a, q_b, log_p_a, log_p_b, U):
        sample = U * (q_b - q_a)
        return sample

    sample = jax.lax.cond(
        is_flat(q_a, q_b, log_p_a, log_p_b),
        flat_dist,
        not_flat_dist,
        q_a,
        q_b,
        log_p_a,
        log_p_b,
        U,
    )

    return q_a + sample


def log_likelihood_regular(
    x: float,
    quantiles: Array,
    logits_full: Array,
    log_mus: Tuple[float, float],
) -> float:
    """Likelihood in a regular bucket."""

    n = len(quantiles) - 1
    assert len(logits_full) == n + 3

    idx = 1 + jnp.searchsorted(quantiles[1:-1], x)

    q_a, q_b = quantiles[idx - 1], quantiles[idx]
    log_p_a, log_p_b = logits_full[idx], logits_full[idx + 1]

    def flat_dist(q_a, q_b, log_p_a, log_p_b, x):
        return log_p_a

    def not_flat_dist(q_a, q_b, log_p_a, log_p_b, x):
        delta = x - q_a

        (log_abs_a, sgn_a), log_b = _make_log_dist_params(
            q_a=q_a, q_b=q_b, log_p_a=log_p_a, log_p_b=log_p_b
        )

        x_log_prob, sgn_x = jax.scipy.special.logsumexp(
            jnp.array([log_abs_a, log_b]),
            b=jnp.array([sgn_a * delta, 1]),
            return_sign=True,
        )

        x_log_prob = jax.lax.cond(
            jnp.logical_and(sgn_x > 0, x_log_prob > LOG_EPS),
            lambda x: x,
            lambda _: LOG_EPS,
            x_log_prob,
        )
        return x_log_prob

    def x_on_edge(q_a, q_b, log_p_a, log_p_b, x):
        return jax.lax.cond(x - q_a < EPS, lambda _: log_p_a, lambda _: log_p_b, None)

    def x_not_on_edge(q_a, q_b, log_p_a, log_p_b, x):
        x_log_prob = jax.lax.cond(
            is_flat(q_a, q_b, log_p_a, log_p_b),
            flat_dist,
            not_flat_dist,
            q_a,
            q_b,
            log_p_a,
            log_p_b,
            x,
        )
        return x_log_prob

    x_log_prob = jax.lax.cond(
        jnp.logical_or(x - q_a < EPS, q_b - x < EPS),
        x_on_edge,
        x_not_on_edge,
        q_a,
        q_b,
        log_p_a,
        log_p_b,
        x,
    )
    return x_log_prob
