from __future__ import annotations
import random
import dataclasses

import jax
import jax.numpy as jnp
import optax

import typing
from flax.core.scope import VariableDict
from flax.metrics.tensorboard import SummaryWriter
from gallimimus.model import MetaLearner
from gallimimus.codec.abstract_codec import Observation
from gallimimus.training.configs import TrainingConfig
from gallimimus.training.optimizer import get_optimizers
from gallimimus.training.logging_utils import _log_global_training_info,_log_step_training_info

def tree_transpose(list_of_trees: typing.List[typing.Any]):
    """Convert a list of trees of identical structure into a single tree of lists."""
    trees_stacked = jax.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)
    return trees_stacked


def train(
    model: MetaLearner,
    model_params: VariableDict,
    dataset: typing.Iterator[typing.List[Observation]],
    training_config: TrainingConfig,
) -> typing.Tuple[VariableDict,VariableDict,VariableDict]:
    """Train the model.

    :param model: A model to be trained.
    :param params: The flax parameters at initialization.
    :param hyperparams: Configuration for the training.
    :param dataset: A list of observations to train on.
    :param optimizer_seed: Starting seed for the noise in DP-SGD training.
    :return: The parameters after training the model according to the hyperparameters.
    """
    rng = jax.random.PRNGKey(training_config.random_seed)
    standard_optimizer, dp_optimizer = get_optimizers(
        num_train_steps=training_config.num_train_steps,
        optimizer_config=training_config.optimizer_config,
    )

    #TODO: Allow loading existing optim state
    opt_state = standard_optimizer.init(model_params)

    if training_config.optimizer_config.is_dp:
        dp_state = dp_optimizer.init(model_params)
        #apply loss per gradient
        apply_fn = model.loss_and_per_example_grad
    else:
        # sgd takes the regular batch gradients
        apply_fn = model.loss_and_grad

    # define and compile the update step:
    @jax.jit
    def train_step(params:VariableDict, opt_state,dp_state, inputs,is_dp:bool)->typing.Tuple:
        loss, grads = apply_fn(params, inputs)
        if is_dp:
            grads, dp_state = dp_optimizer.update(grads, dp_state)
        updates, opt_state = standard_optimizer.update(
            grads, opt_state, params=params
        )
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state, dp_state

    loss_accumulation = []
    logged_losses = []
    step_with_updates=0
    for step in range(1, training_config.num_train_steps + 1):
        batch = next(dataset)
        rng, _ = jax.random.split(rng)
        batch_loss, model_params, opt_state, dp_state = train_step(
            model_params,
            opt_state,
            dp_state,
            batch,
            rng,
        )
        loss_accumulation.append(batch_loss.mean())
        if step_with_updates % training_config.check_point_config.logging_steps == 0:
            loss_to_log = sum(loss_accumulation) / len(loss_accumulation)
            _log_step_training_info(step=step_with_updates, loss=loss_to_log)
            logged_losses.append(loss_to_log)
            loss_accumulation = []
        if step_with_updates % training_config.check_point_config.save_every_steps == 0:
            #TODO: IMPLEMENT CHECKPOINT
            pass
        if step%training_config.optimizer_config.gradient_accumulation_step==0:
            step_with_updates += 1
    return model_params,opt_state,dp_state

