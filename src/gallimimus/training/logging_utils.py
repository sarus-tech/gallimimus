import logging

import jax
import optax

from sarus_synthetic_data.jax_model.model import MetaLearner
from sarus_synthetic_data.jax_model.training.configs import (
    CheckpointConfig,
    TrainingConfig,
)
from sarus_synthetic_data.jax_model.training.optimizer import (
    save_optimizer_state,
)
from sarus_synthetic_data.jax_model.typing import ModelParams

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


def _log_step_training_info(step: int, loss: int) -> None:

    logger.info(f"***** Training step {step} *****")
    logger.info(f"Current Loss = {loss}")
