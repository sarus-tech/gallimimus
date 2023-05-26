from __future__ import annotations
import random
import dataclasses

import jax
import jax.numpy as jnp
import optax

from typing import List, Any
from flax.core.scope import VariableDict

from gallimimus.model import MetaLearner
from gallimimus.codec.abstract_codec import Observation


def tree_transpose(list_of_trees: List[Any]):
    """Convert a list of trees of identical structure into a single tree of lists."""
    trees_stacked = jax.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)
    return trees_stacked


@dataclasses.dataclass
class TrainingHyperparameters:
    """Parameters for the training."""

    num_epochs: int = 10
    """Number of training epochs."""
    batch_size: int = 128
    """Batch size."""
    learning_rate: float = 1.0
    """Learning rate of the optimizer."""

    dp: bool = False
    """Trains with SGD if ``False``, DP-SGD if ``True``."""
    noise_multiplier: float = 0.0
    """Noise multiplier for the DP-SGD."""
    l2_norm_clip: float = 1e10
    """Norm of the gradient clipping for the DP-SGD."""


def train(
    model: MetaLearner,
    params: VariableDict,
    hyperparams: TrainingHyperparameters,
    dataset: List[Observation],  # List of observations
    optimizer_seed: int = 0,
) -> VariableDict:
    """Train the model.

    :param model: A model to be trained.
    :param params: The flax parameters at initialization.
    :param hyperparams: Configuration for the training.
    :param dataset: A list of observations to train on.
    :param optimizer_seed: Starting seed for the noise in DP-SGD training.
    :return: The parameters after training the model according to the hyperparameters.
    """
    # select the optimizer and loss function:
    if hyperparams.dp:
        tx = optax.chain(
            optax.differentially_private_aggregate(
                l2_norm_clip=hyperparams.l2_norm_clip,
                noise_multiplier=hyperparams.noise_multiplier,
                seed=optimizer_seed,
            ),
            optax.sgd(learning_rate=hyperparams.learning_rate),
        )
        # `differentially_private_aggregate` takes the per-instance gradients
        apply_fn = model.loss_and_per_example_grad
    else:
        tx = optax.sgd(
            learning_rate=hyperparams.learning_rate,
        )
        # sgd takes the regular batch gradients
        apply_fn = model.loss_and_grad

    # define and compile the update step:
    @jax.jit
    def train_step(params, opt_state, inputs):
        loss, grads = apply_fn(params, inputs)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # training loop
    bs = hyperparams.batch_size
    n_batches = len(dataset) // bs

    opt_state = tx.init(params)

    try:
        for i in range(hyperparams.num_epochs):
            dataset_i = dataset[:]  # copy the dataset
            random.Random(i).shuffle(dataset_i)

            losses = []
            for j in range(n_batches):
                batch_list = dataset[j * bs : (j + 1) * bs]
                batch = tree_transpose(batch_list)

                params, opt_state, loss_val = train_step(params, opt_state, batch)
                losses.append(loss_val)

            loss_epoch = float(jnp.sum(jnp.array(losses)))
            print(f"{i}: {loss_epoch=}")

        return params

    except KeyboardInterrupt:
        return params
