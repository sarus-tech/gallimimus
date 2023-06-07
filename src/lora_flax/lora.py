import typing as t
from collections.abc import Mapping

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
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


def lora_combine_vars(
    pretrained_var: jax.Array,
    lora_vars: t.Tuple[jax.Array, jax.Array],
    alpha: float = 1.0,
):
    A, B = lora_vars
    return pretrained_var + alpha * A @ B


def lora_combine_params(
    pretrained_params: VariableDict,
    lora_params: VariableDict,
    alpha: float = 1.0,
) -> VariableDict:
    """Adds the LoRA fine-tuning to the pretrained parameters.

    :param pretrained_params: A PyTree containing the original parameters
    :param lora_params: A PyTree which is a subset of ``pretrained_params`` and contains the trained low-rank perturbations to add
    :param alpha: scaling of the perturbation in the sum
    :return: A PyTree containing the updated parameters (same shape as ``pretrained_params``)
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


def lora_combine_params2(
    pretrained_params: VariableDict,
    lora_params: VariableDict,
    alpha: float = 1.0,
):
    """Adds the LoRA fine-tuning to the pretrained parameters.

    :param pretrained_params: A PyTree containing the original parameters
    :param lora_params: A PyTree which is a subset of ``pretrained_params`` and contains the trained low-rank perturbations to add
    :param alpha: scaling of the perturbation in the sum
    :return: A PyTree containing the updated parameters (same shape as ``pretrained_params``)
    """
    res = {}
    for k, v in pretrained_params.items():
        if k not in lora_params:
            res[k] = v
        else:
            if isinstance(v, Mapping):
                res[k] = lora_combine_params2(v, lora_params[k], alpha)
            else:
                res[k] = lora_combine_vars(v, lora_params[k], alpha)
    return res


def LoRA(
    target_module: nn.Module,
    pretrained_params,
    filter_fn,
    r: int,
    methods: t.Dict = {"__call__": []},
) -> nn.Module:
    """
    TODO
    """

    class LoRA_module(nn.Module):
        """Wrapper module enabling `LoRa approximation <https://arxiv.org/abs/2106.09685>`_"""

        def setup(self):
            """TODO"""
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
                {"params": summed_params}, *args, **kwargs, rngs=rngs, method=method_name
            )

            return output

        return method_fn

    for method_name, need_rng in methods.items():
        setattr(LoRA_module, method_name, _get_method_fn(method_name, need_rng))

    return LoRA_module()
