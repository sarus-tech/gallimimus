from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import jax.numpy as jnp
import optax

logger = logging.getLogger(__name__)

OPTIMIZER_NAME = "opt_state.msgpack"


@dataclass
class OptimizerConfig:
    """Config to specify the optimizer, it wraps:

    - whether dp or not
    - whether to have a decayed learning rate
    - the type: adam or sgs
    - whether to accumulate gradients
    - whether to load an existing optimizer
    """

    is_dp: bool = field(
        metadata={"help": "Whether to train with Differential Privacy or not"}
    )
    clipping_norm: float = field(
        default=1e-2,
        metadata={"help": "Clipping norm for per_sample gradient"},
    )
    noise_multiplier: float = field(
        default=0.0, metadata={"help": "Noise std to add at each iteration"}
    )
    optim: str = field(
        default="adam",
        metadata={"help": 'The optimizer to use. Can be "adam" (default) or "sgd"'},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Number of updates steps to accumulate before performing an"
                " update pass."
            )
        },
    )

    learning_rate: float = field(
        default=1e-2, metadata={"help": "The initial learning rate."}
    )

    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay applied to parameters."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for Adam "},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for for Adam"},
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for Adam optimizer."}
    )
    lr_decay: str = field(
        default=None,
        metadata={
            "help": (
                "Decay to be used in the learning rate scheduler. Can be None"
                " (default), linear or exponential."
            )
        },
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )
    lr_transition_steps: int = field(
        default=None,
        metadata={
            "help": (
                "Number of transition steps associated with learning rate"
                " decay when using exponential decay."
            )
        },
    )
    lr_decay_rate: float = field(
        default=None,
        metadata={
            "help": (
                "Decay rate associated with learning rate when using exponential decay."
            )
        },
    )
    lr_staircase: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use staircase or continuous learning rate when"
                " using exponential decay."
            )
        },
    )
    lr_offset: int = field(
        default=0,
        metadata={"help": "Number of steps to offset learning rate and keep it at 0."},
    )
    load_state: bool = field(
        default=False,
        metadata={"help": "Whether to reuse an existing optimizer state"},
    )
    state_dir: str = field(
        default="",
        metadata={"help": "Directory where to load the optimizer if it exist"},
    )
    dp_init_seed: int = field(
        default=0,
        metadata={"help": "Seed to init state of dp optimizer for noise"},
    )


def _create_learning_rate_fn(
    num_train_steps: int, optimizer_args: OptimizerConfig
) -> t.Callable[[int], jnp.array]:
    """Create the learning rate function.

    It consists in a constant learning rate followed by a decay
    """

    warmup_fn = optax.linear_schedule(
        init_value=optimizer_args.learning_rate,
        end_value=optimizer_args.learning_rate,
        transition_steps=optimizer_args.warmup_steps + 1,  # ensure not 0
    )
    last_boundary = optimizer_args.warmup_steps
    # offset step when resuming
    if optimizer_args.lr_offset:
        warmup_fn = optax.join_schedules(
            schedules=[optax.constant_schedule(0.0), warmup_fn],
            boundaries=[optimizer_args.lr_offset],
        )
        last_boundary += optimizer_args.lr_offset
    if optimizer_args.lr_decay is None:
        return warmup_fn
    elif optimizer_args.lr_decay == "linear":
        assert (
            num_train_steps is not None
        ), "linear decay requires knowing the dataset length"
        decay_fn = optax.linear_schedule(
            init_value=optimizer_args.learning_rate,
            end_value=0,
            transition_steps=num_train_steps - optimizer_args.warmup_steps,
        )
    elif optimizer_args.lr_decay == "exponential":
        decay_fn = optax.exponential_decay(
            init_value=optimizer_args.learning_rate,
            transition_steps=optimizer_args.lr_transition_steps,
            decay_rate=optimizer_args.lr_decay_rate,
            staircase=optimizer_args.lr_staircase,
        )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[last_boundary],
    )
    return schedule_fn


def get_optimizers(
    num_train_steps: int, optimizer_config: OptimizerConfig
) -> t.Tuple[optax.GradientTransformation, optax.GradientTransformation]:
    """Method to retrieve the optimizers from the config.

    We currently use two optimizers: one to apply dp aggregation to
    the gradients and one for the rest (lr scaling+ aggregation).
    This is necessary because the dp aggregation changes the shape
    of the gradients and the gradient aggregator wrapper uses the shape
    to add zero gradients when it is accumulating them. So it forbids
    having input gradients of shape different from the output.
    """

    learning_rate_fn = _create_learning_rate_fn(
        num_train_steps=num_train_steps, optimizer_args=optimizer_config
    )
    if optimizer_config.optim == "adam":
        optimizer = optax.adamw(
            learning_rate=learning_rate_fn,
            b1=optimizer_config.beta1,
            b2=optimizer_config.beta2,
            eps=optimizer_config.adam_epsilon,
            weight_decay=optimizer_config.weight_decay,
        )
    else:
        optimizer = optax.sgd(learning_rate=learning_rate_fn)

    if optimizer_config.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(
            opt=optimizer,
            every_k_schedule=optimizer_config.gradient_accumulation_steps,
        )

    return optimizer, optax.differentially_private_aggregate(
        l2_norm_clip=optimizer_config.clipping_norm,
        noise_multiplier=optimizer_config.noise_multiplier,
        seed=optimizer_config.dp_init_seed,
    )
