from __future__ import annotations

import typing

import flax.struct as struct
import jax
import jax.numpy as jnp
import optax
from flax.core.scope import VariableDict
from flax.metrics.tensorboard import SummaryWriter

# from flax.metrics.tensorboard import SummaryWriter
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

from gallimimus.codec.abstract_codec import Observation
from gallimimus.model import MetaLearner
from gallimimus.training.configs import TrainingConfig
from gallimimus.training.logging_utils import _log_step_training_info
from gallimimus.training.optimizer import get_optimizers


def tree_transpose(list_of_trees: typing.List[typing.Any]):
    """Convert a list of trees of identical structure into a single tree of lists."""
    trees_stacked = jax.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)
    return trees_stacked


@struct.dataclass
class TrainState:
    model_params: VariableDict
    dp_state: typing.Any
    standard_state: typing.Any


def train(
    model: MetaLearner,
    model_params: VariableDict,
    dataset: typing.Iterator[typing.List[Observation]],
    eval_set: typing.Iterator[typing.List[Observation]],
    training_config: TrainingConfig,
    standard_state: typing.Optional[VariableDict],
    dp_state: typing.Optional[VariableDict],
    shift_step: int,
) -> typing.Tuple[VariableDict, VariableDict, VariableDict]:
    """Train the model.

    :param model: A model to be trained.
    :param params: The flax parameters at initialization.
    :param dataset: An iterator of observations to train on. :training_config
    :return: The parameters after training the model according to the hyperparameters.
    """
    jax.random.PRNGKey(training_config.random_seed)
    standard_optimizer, dp_optimizer = get_optimizers(
        num_train_steps=training_config.num_train_steps,
        optimizer_config=training_config.optimizer_config,
    )
    if standard_state is None:
        standard_state = standard_optimizer.init(model_params)
    if dp_state is None:
        dp_state = dp_optimizer.init(model_params)
    summary_writer = SummaryWriter(training_config.check_point_config.tensorboard_dir)

    # define and compile the update step:
    def standard_train_step(
        params: VariableDict, opt_state, dp_state, inputs
    ) -> typing.Tuple:
        _, grads, loss_tree = model.loss_and_grad(params, inputs)
        updates, standard_state = standard_optimizer.update(
            grads, opt_state, params=params
        )
        params = optax.apply_updates(params, updates)
        return (
            jax.tree_map(lambda x: x.mean(), loss_tree),
            params,
            standard_state,
            dp_state,
        )

    def dp_train_step(
        params: VariableDict, opt_state, dp_state, inputs
    ) -> typing.Tuple:
        loss, grads, loss_tree = model.loss_and_per_example_grad(params, inputs)
        grads, dp_state = dp_optimizer.update(grads, dp_state)
        updates, opt_state = standard_optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state, dp_state

    if training_config.optimizer_config.is_dp:
        # apply loss per gradient
        train_step = jax.jit(dp_train_step)
    else:
        # sgd takes the regular batch gradients
        train_step = jax.jit(standard_train_step)

    loss_func = jax.jit(model.batch_loss)

    loss_accumulation = []
    logged_losses = []
    options = CheckpointManagerOptions(max_to_keep=3, keep_period=2)
    mngr = CheckpointManager(
        training_config.check_point_config.output_dir,
        PyTreeCheckpointer(),
        options=options,
    )
    for step in range(1, training_config.num_train_steps + 1):
        batch = next(dataset)
        batch_loss, model_params, standard_state, dp_state = train_step(
            model_params, standard_state, dp_state, batch
        )
        loss_accumulation.append(batch_loss)
        if (step + shift_step) % training_config.check_point_config.logging_steps == 0:
            loss_to_log = sum(jax.tree_util.tree_flatten(loss_accumulation)[0]) / len(
                loss_accumulation
            )
            _log_step_training_info(step=shift_step + step, loss=loss_to_log)
            logged_losses.append(loss_to_log)

            reorganised_loss = jax.tree_util.tree_transpose(
                outer_treedef=jax.tree_util.tree_structure(
                    [0 for e in loss_accumulation]
                ),
                inner_treedef=jax.tree_util.tree_structure(loss_accumulation[0]),
                pytree_to_transpose=loss_accumulation,
            )
            summary_writer.scalar("train_loss", loss_to_log, step=step + shift_step)
            for key, vals in reorganised_loss.items():
                summary_writer.scalar(
                    "train_loss_" + key,
                    jnp.stack(vals).mean(),
                    step=step + shift_step,
                )

            loss_accumulation = []

        if (
            step + shift_step
        ) % training_config.check_point_config.save_every_steps == 0:
            train_state = TrainState(model_params, dp_state, standard_state)
            mngr.save(step + shift_step, train_state)

        if (
            training_config.eval_every_step is not None
            and (shift_step + step) % training_config.eval_every_step == 0
        ):
            eval_losses = []
            for eval_batch in eval_set():
                eval_losses.append(
                    jax.tree_map(
                        lambda x: x.mean(),
                        loss_func(model_params, eval_batch)[1],
                    )
                )

            eval_loss = sum(jax.tree_util.tree_flatten(eval_losses)[0]) / len(
                eval_losses
            )
            _log_step_training_info(
                step=step + shift_step, loss=eval_loss, is_training=False
            )
            summary_writer.scalar("eval_loss", eval_loss, step=step + shift_step)

            reorganised_loss = jax.tree_util.tree_transpose(
                outer_treedef=jax.tree_util.tree_structure([0 for e in eval_losses]),
                inner_treedef=jax.tree_util.tree_structure(eval_losses[0]),
                pytree_to_transpose=eval_losses,
            )
            for key, vals in reorganised_loss.items():
                summary_writer.scalar(
                    "eval_loss_" + key,
                    jnp.stack(vals).mean(),
                    step=step + shift_step,
                )

    return model_params, standard_state, dp_state
