from __future__ import annotations
import random
import dataclasses

import jax
import jax.numpy as jnp
import optax
from flax.metrics.tensorboard import SummaryWriter

import typing
from flax.core.scope import VariableDict
#from flax.metrics.tensorboard import SummaryWriter
from orbax.checkpoint import CheckpointManager,CheckpointManagerOptions,PyTreeCheckpointer
from gallimimus.model import MetaLearner
import flax.struct as struct
from gallimimus.codec.abstract_codec import Observation
from gallimimus.training.configs import TrainingConfig
from gallimimus.training.optimizer import get_optimizers
from gallimimus.training.logging_utils import _log_global_training_info,_log_step_training_info

def tree_transpose(list_of_trees: typing.List[typing.Any]):
    """Convert a list of trees of identical structure into a single tree of lists."""
    trees_stacked = jax.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)
    return trees_stacked

@struct.dataclass
class TrainState:
    model_params:VariableDict
    dp_state:typing.Any
    standard_state:Any

def train(
    model: MetaLearner,
    model_params: VariableDict,
    dataset: typing.Iterator[typing.List[Observation]],
    eval_set:typing.Iterator[typing.List[Observation]],
    training_config: TrainingConfig,
) -> typing.Tuple[VariableDict,VariableDict,VariableDict]:
    """Train the model.

    :param model: A model to be trained.
    :param params: The flax parameters at initialization.
    :param dataset: An iterator of observations to train on.
    :training_config
    :return: The parameters after training the model according to the hyperparameters.
    """
    rng = jax.random.PRNGKey(training_config.random_seed)
    standard_optimizer, dp_optimizer = get_optimizers(
        num_train_steps=training_config.num_train_steps,
        optimizer_config=training_config.optimizer_config,
    )

    #TODO: Allow loading existing optim state
    opt_state = standard_optimizer.init(model_params)
    dp_state = dp_optimizer.init(model_params)
    summary_writer=SummaryWriter(training_config.check_point_config.tensorboard_dir)

    # define and compile the update step:
    def standard_train_step(params:VariableDict, opt_state,dp_state, inputs)->typing.Tuple:
        loss, grads = model.loss_and_grad(params, inputs)
        updates, opt_state = standard_optimizer.update(
            grads, opt_state, params=params
        )
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state, dp_state

    def dp_train_step(params:VariableDict, opt_state,dp_state, inputs)->typing.Tuple:
        loss, grads = model.loss_and_per_example_grad(params, inputs)
        grads, dp_state = dp_optimizer.update(grads, dp_state)
        updates, opt_state = standard_optimizer.update(
            grads, opt_state, params=params
        )
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state, dp_state

    if training_config.optimizer_config.is_dp:
        # apply loss per gradient
        train_step = jax.jit(dp_train_step)
    else:
        # sgd takes the regular batch gradients
        train_step = jax.jit(standard_train_step)
    
    loss_func=jax.jit(model.batch_loss)
    
    loss_accumulation = []
    logged_losses = []
    step_with_updates=1
    options = CheckpointManagerOptions(max_to_keep=3, keep_period=2)
    mngr = CheckpointManager(
        training_config.check_point_config.output_dir, PyTreeCheckpointer(),
        options=options)
    for step in range(1, training_config.num_train_steps + 1):
        batch = next(dataset)
        batch_loss, model_params, opt_state, dp_state = train_step(
            model_params,
            opt_state,
            dp_state,
            batch
        )
        loss_accumulation.append(batch_loss.mean())
        if step_with_updates % training_config.check_point_config.logging_steps == 0:
            loss_to_log = sum(loss_accumulation) / len(loss_accumulation)
            _log_step_training_info(step=step_with_updates, loss=loss_to_log)
            logged_losses.append(loss_to_log)
            loss_accumulation = []
            summary_writer.scalar('train_loss',loss_to_log,step=step_with_updates)
        if step_with_updates % training_config.check_point_config.save_every_steps == 0:
            train_state=TrainState(model_params,opt_state,dp_state)
            mngr.save(step_with_updates, train_state)

        if training_config.eval_every_step is not None and step_with_updates%training_config.eval_every_step == 0:
            eval_losses=[]
            for eval_batch in eval_set():
                eval_losses.append(loss_func(model_params,eval_batch,rng))
            
            _log_step_training_info(step=step_with_updates, loss=sum(eval_losses) / len(eval_losses),is_training=False)
            summary_writer.scalar('eval_loss',sum(eval_losses) / len(eval_losses),step=step_with_updates)
        if step%training_config.optimizer_config.gradient_accumulation_steps==0:
            step_with_updates += 1
    return model_params,opt_state,dp_state

