from collections.abc import Mapping

import jax.numpy as jnp
import jax
import flax
import flax.linen as nn
import typing as t

from flax.core.scope import VariableDict
from jax.random import KeyArray

FilterFunction = t.Callable[[t.List[str], jax.Array], bool]


def init_lora_params(
    rng: KeyArray,
    params: VariableDict,
    filter_fn: FilterFunction,
    r: int,
) -> VariableDict:
    """Initialize the PyTree containing the low-rank perturbations.

    :param rng: a jax PRNGKey
    :param params: the PyTree of the pretrained parameters
    :param filter_fn: A function of signature ``(param_name: t.List[str], param: jax.Array) -> bool`` deciding if a parameter is finetuned.
    :param r: The rank of the LoRA approximation.
    :return: A PyTree which is a subset of ``pretrained_params`` and contains the initialized low-rank perturbations
    """
    params_flat = flax.traverse_util.flatten_dict(
        params,
    )

    rngs = jax.random.split(key=rng, num=len(params_flat))

    def _init_lora_params(param, rng):
        shape = param.shape
        assert len(shape) == 2

        a, b = shape[0], shape[-1]
        A = jax.random.normal(key=rng, shape=(a, r))
        B = jnp.zeros(shape=(r, b))
        return A, B

    lora_params_flat = {
        param_name: _init_lora_params(param, rng)
        for rng, (param_name, param) in zip(rngs, params_flat.items())
        if filter_fn(param_name, param)
    }
    lora_params = flax.traverse_util.unflatten_dict(lora_params_flat)
    return lora_params


def lora_combine_params(
    pretrained_params: VariableDict,
    lora_params: VariableDict,
    alpha: float = 1.0,
) -> VariableDict:
    """
    Adds the LoRA fine-tuning to the pretrained parameters.

    :param pretrained_params: A PyTree containing the original parameters
    :param lora_params: A PyTree which is a subset of ``pretrained_params`` and contains the trained low-rank perturbations to add
    :param alpha: scaling of the perturbation in the sum
    :return: A PyTree containing the updated parameters (same shape as ``pretrained_params``)
    """
    lora_params_flat = flax.traverse_util.flatten_dict(
        lora_params,
    )
    pretrained_params_flat = flax.traverse_util.flatten_dict(pretrained_params)

    def _lora_combine_params(pretrained_param, lora_param):
        (left, right) = lora_param
        return pretrained_param + alpha * left @ right

    total_flat = {
        k: (
            _lora_combine_params(pretrained_param, lora_params_flat[k])
            if k in lora_params_flat
            else pretrained_param
        )
        for k, pretrained_param in pretrained_params_flat.items()
    }

    return flax.traverse_util.unflatten_dict(total_flat)


class LoRA(nn.Module):
    """Wrapper module enabling `LoRa approximation <https://arxiv.org/abs/2106.09685>`_"""

    target_apply_fn: t.Callable
    """The ``nn.Module.apply`` function of the module LoRA is applied to."""
    pretrained_params: VariableDict
    """The pretrained parameters of the model."""

    filter_fn: FilterFunction
    """A function of signature ``(param_name: t.List[str], param: jax.Array) -> bool`` deciding if a parameter is finetuned."""
    r: int
    """The rank of the LoRA approximation."""

    @nn.compact
    def __call__(self, *args, **kwargs):
        """Calls the ``target_apply_fn``. The signature is the same as ``target_apply_fn``."""
        lora_params = self.param(
            "lora",
            init_lora_params,
            self.pretrained_params,
            self.filter_fn,
            self.r,
        )

        summed_params = lora_combine_params(
            pretrained_params=self.pretrained_params,
            lora_params=lora_params,
        )

        return self.target_apply_fn({"params": summed_params}, *args, **kwargs)
