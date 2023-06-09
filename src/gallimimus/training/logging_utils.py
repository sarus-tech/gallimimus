import logging

import jax
import optax
from gallimimus.training.configs import (
    CheckpointConfig,
    TrainingConfig,
)


logger = logging.getLogger(__name__)


def _log_global_training_info(
    training_config: TrainingConfig, n_params: int
) -> None:

    logger.info("***** Running training *****")
    logger.info(f"  Num Training Steps = {training_config.num_train_steps}")
    logger.info(
        "  Gradient accumulation steps ="
        f" {training_config.optimizer_config.gradient_accumulation_steps}"
    )
    logger.info(f"  Batch size per step {training_config.batch_size}")
    logger.info(f"  Number of devices = {jax.device_count()}")
    logger.info(f"  Model parameters = {n_params}")


def _log_step_training_info(step: int, loss: int,is_training:bool=True) -> None:

    train= 'Train 'if is_training else 'Test'
    logger.info(f"***** Training step {step} *****")
    logger.info(f"Current {train} Loss = {loss}")
