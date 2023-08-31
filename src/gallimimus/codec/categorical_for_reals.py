from __future__ import annotations

from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from gallimimus.codec.abstract_codec import Codec, Embedding
from gallimimus.codec.categorical_for_reals_utils import (
    interpolate_generic as interpolate_generic,
)
from gallimimus.shared_codecs import SharedCodecs

Array = Any

RealCategoricalObservation = jax.Array  # of shape () and dtype float
RealCategoricalContext = None
RealCategoricalPrediction = jax.Array  # un-normalized logits of shape (vocab_size,)


class RealCategoricalCodec(Codec):
    """Feature for real numbers.

    Uses quantiles as categories to represent the distribution
    """

    interpolate_mode: int

    n_quantiles: Optional[int] = None

    quantiles: Optional[Array] = None
    data: Optional[Array] = None

    # tail_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    # TODO unused

    do_tails: Tuple[bool, bool] = (True, True)

    def make_quantiles(self):
        """Defines the quantiles from the provided data."""

        assert self.data is not None and self.n_quantiles is not None

        # warning because of 'unique' might give a quantile array of len < n_quantiles
        self.quantiles = jnp.unique(
            jnp.quantile(
                self.data,
                q=jnp.linspace(0, 1, self.n_quantiles + 1, endpoint=True),
                method="linear",
            )
        )

    def __post_init__(self):
        # this function is run at instantiation
        if self.quantiles is None:
            self.make_quantiles()

        self.n_quantiles = len(self.quantiles) - 1
        del self.data

    def setup(self):
        self.embed = nn.Embed(self.feature.n_quantiles + 3, self.embed_dim)
        # 0 : left_tail embedding
        # 1:(n+2) : quantile embeddings
        # n+2 : right_tail embedding

        # self.log_mus = self.param("log_mus", lambda rng: jnp.array([1.0, 1.0]))

    def _choose_tail_value(self, which_tail, val, default_val=None):
        # returns the value if we want to have a decaying tail on this side,
        # otherwise returns None (or the provided default_val)
        return jax.lax.cond(
            self.feature.do_tails[which_tail],
            lambda _: val,
            lambda _: default_val,
            None,
        )

    def encode(self, x: RealCategoricalObservation, shared_codecs: SharedCodecs):
        assert x.shape == ()

        embedding = interpolate_generic.embed_full(
            interpolate_mode=self.interpolate_mode,
            inputs=x,
            quantiles=self.quantiles,
            embeddings=self.embed.embedding,
        )

        return embedding, None

    def decode(
        self,
        conditioning_vector: Embedding,
        context: RealCategoricalContext,
        shared_codecs: SharedCodecs,
    ) -> RealCategoricalPrediction:
        """The second argument is unused here, added to respect the Codec
        specification."""

        assert conditioning_vector.shape == (self.embed_dim,)

        logits_full_with_tails = self.embed.attend(conditioning_vector)
        logits_q = logits_full_with_tails[1:-1]

        # the logits for the tails are set to minus infinity
        # if we do not want a decaying tail on this side
        logits_l = self._choose_tail_value(
            which_tail=0,
            val=logits_full_with_tails[0],
            default_val=-jnp.inf,
        )
        logits_r = self._choose_tail_value(
            which_tail=1,
            val=logits_full_with_tails[-1],
            default_val=-jnp.inf,
        )

        log_mu_l = self._choose_tail_value(
            which_tail=0,
            val=logits_l - logits_q[0],
            default_val=0.0,
        )
        log_mu_r = self._choose_tail_value(
            which_tail=1,
            val=logits_l - logits_q[-1],
            default_val=0.0,
        )

        logits_full = jnp.concatenate(
            [jnp.array([logits_l]), logits_q, jnp.array([logits_r])]
        )

        log_mus = jnp.array([log_mu_l, log_mu_r])
        return logits_full, (self.quantiles, log_mus)

    def sample(
        self, conditioning_vector: Embedding, shared_codecs: SharedCodecs
    ) -> Tuple[RealCategoricalObservation, Embedding]:
        rng = self.make_rng("sample")

        logits_full, (quantiles, log_mus) = self.decode(conditioning_vector, None)

        sample = interpolate_generic.sample_full(
            interpolate_mode=self.interpolate_mode,
            rng=rng,
            logits_full=logits_full,
            quantiles=quantiles,
            log_mus=log_mus,
        )

        return sample, self.encode(sample)[0]

    def marginal(self, path: Tuple[str], context: jnp.ndarray):
        # TODO not implemented
        raise NotImplementedError

        assert path[0] == self.feature.name
        assert len(path) == 1
        return nn.softmax(jnp.dot(context, self.embed.embedding.T))

    def embedding(self, path: Tuple[str]):
        # TODO not implemented
        raise NotImplementedError

        assert len(path) == 1
        assert path[0] == self.feature.name
        return self.embed.embedding

    def loss(
        self,
        x: RealCategoricalObservation,
        prediction: RealCategoricalPrediction,
        shared_codecs: SharedCodecs,
    ) -> jnp.ndarray:
        logits_full, (quantiles, log_mus) = prediction

        assert logits_full.shape == (len(quantiles) + 2,)
        assert x.shape == ()

        log_likelihood = interpolate_generic.log_likelihood(
            interpolate_mode=self.interpolate_mode,
            x=x,
            logits_full_unn=logits_full,
            quantiles=quantiles,
            log_mus=log_mus,
        )
        return -log_likelihood
