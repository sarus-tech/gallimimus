from __future__ import annotations

from dataclasses import dataclass, field

from gallimimus.training.optimizer import OptimizerConfig


@dataclass
class TrainingConfig:
    """General config during training"""

    random_seed: int
    optimizer_config: OptimizerConfig
    check_point_config: CheckpointConfig
    num_train_steps: int = field(
        metadata={"help": "Total number of training steps to perform."}
    )
    batch_size: int = field(metadata={"help": "Batch Size per training step."})
    params_dtype: str = field(
        default="float32",
        metadata={"help": "Dtype of params to reduce memory"},
    )
    eval_every_step:int=field(default=None,metadata={"help": "Evaluate on test set every tot updates"})
    


@dataclass
class CheckpointConfig:
    """Config for saving state (model params+optim) during training."""

    output_dir: str = field(
        metadata={
            "help": (
                "The output directory where the model predictions and"
                " checkpoints will be written."
            )
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. Use this to"
                " continue training if output_dir points to a checkpoint"
                " directory."
            )
        },
    )
    logging_steps: int = field(
        default=1, metadata={"help": "Log every X updates steps."}
    )
    save_every_steps: int = field(
        default=5, metadata={"help": "Save checkpoint every X updates steps."}
    )
    tensorboard_dir:str=field(
        default='', metadata={"help": "Save checkpoint every X updates steps."}
    )