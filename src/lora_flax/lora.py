from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.scope import VariableDict
from jax.random import KeyArray

FilterFunction = Callable[[List[str], jax.Array], bool]


def init_lora_params(
    rng: KeyArray,
    params: VariableDict,
    filter_fn: FilterFunction,
    r: int,
) -> VariableDict:
    """Initialize the PyTree containing the low-rank perturbations for params of a
    certain shape.

    :param rng: a jax PRNGKey
    :param params: the PyTree of the pretrained parameters
    :param filter_fn: A function of signature ``(param_name: t.List[str],
        param: jax.Array) -> bool`` deciding if a parameter is finetuned.
    :param r: The rank of the LoRA approximation.
    :return: A PyTree which is a subset of ``pretrained_params`` and contains the
        initialized low-rank perturbations
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


def lora_combine_vars(
    pretrained_var: jax.Array,
    lora_vars: Tuple[jax.Array, jax.Array],
    alpha: float = 1.0,
):
    A, B = lora_vars
    return pretrained_var + alpha * A @ B


def lora_combine_params(
    pretrained_params: VariableDict,
    lora_params: VariableDict,
    alpha: float = 1.0,
) -> VariableDict:
    """Sums the LoRA fine-tuning to the pretrained parameters.

    :param pretrained_params: A PyTree containing the original parameters
    :param lora_params: A PyTree which is a subset of ``pretrained_params`` and
        contains the trained low-rank perturbations to add
    :param alpha: scaling of the perturbation in the sum
    :return: A PyTree containing the updated parameters (same shape as
        ``pretrained_params``)
    """
    lora_params_flat = flax.traverse_util.flatten_dict(
        lora_params,
    )
    pretrained_params_flat = flax.traverse_util.flatten_dict(pretrained_params)

    total_flat = {
        k: (
            lora_combine_vars(pretrained_param, lora_params_flat[k], alpha)
            if k in lora_params_flat
            else pretrained_param
        )
        for k, pretrained_param in pretrained_params_flat.items()
    }

    return flax.traverse_util.unflatten_dict(total_flat)


def LoRA(
    target_module: nn.Module,
    pretrained_params: VariableDict,
    filter_fn,
    r: int,
    methods: Optional[Dict[str, List[str]]] = None,
) -> nn.Module:
    """
    Builds a Flax module that applies  the provided Flax ``target_module`` using the
    ``pretrained_params`` summed with trained LoRA parameters (as described in `this
    article <https://arxiv.org/abs/2106.09685>`_).


    TODO describe the other arguments
    :param methods: a dictionary of all the methods of the Flax module we want to
        transform, each with a list of the names of the ``rngs`` required to apply them.
    """
    if methods is None:
        methods = {"__call__": []}

    class LoRA_module(nn.Module):
        def setup(self):
            self.lora_params = self.param(
                "lora",
                init_lora_params,
                pretrained_params,
                filter_fn,
                r,
            )

    def _get_method_fn(method_name, need_rng):
        def method_fn(self, *args, **kwargs):
            summed_params = lora_combine_params(
                pretrained_params=pretrained_params,
                lora_params=self.lora_params,
            )

            rngs = {rng_name: self.make_rng(rng_name) for rng_name in need_rng}

            output = target_module.apply(
                {"params": summed_params},
                *args,
                **kwargs,
                rngs=rngs,
                method=method_name,
            )

            return output

        return method_fn

    for method_name, need_rng in methods.items():
        setattr(LoRA_module, method_name, _get_method_fn(method_name, need_rng))

    return LoRA_module()
