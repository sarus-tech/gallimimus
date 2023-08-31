import logging
import os
from pathlib import Path

import jax.numpy as jnp
import jax.random
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

from gallimimus import MetaLearner
from gallimimus.codec import CategoricalCodec, LoraCodec, StructCodec
from gallimimus.training.configs import (
    CheckpointConfig,
    OptimizerConfig,
    TrainingConfig,
)
from gallimimus.training.training import train

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

EMBED_DIM = 16
N_TOKENS = 10
MAX_LENGTH = 100
TEXT_MODEL = "distilgpt2"
LORA_RANK = 4
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = TRAIN_BATCH_SIZE
TRAIN_STEPS = 100
GRADS_ACCUM = 1
TRAIN_DP = False

EVAL_EVERY = 15 * GRADS_ACCUM
SAVE_EVERY = 50 * GRADS_ACCUM
TRAIN = True
RESTORE = False
SHIFT_STEP = 0
DO_LORA = False

### dataset

max_cat = 10
N_obs = 1000

dataset = (jnp.arange(N_obs) % max_cat, jnp.arange(N_obs) ** 2 % max_cat)

### Build the model

cat_codec = CategoricalCodec(
    embed_dim=EMBED_DIM,
    vocab_size=max_cat,
)

lora_codec = LoraCodec(
    embed_dim=EMBED_DIM,
    subcodec_in="cat_codec",
    lora_module_name="cat_codec",
    filter_fn=lambda path, arr: len(arr.shape) == 2,
    r=LORA_RANK,
)

struct_codec = StructCodec(
    embed_dim=EMBED_DIM,
    n_heads=8,
    n_blocks=2,
    subcodecs_in=["lora_codec", "lora_codec"],
)

modules_dict = {
    "cat_codec": cat_codec,
    "lora_codec": lora_codec,
    "struct_codec": struct_codec,
}


cat_codec_params = cat_codec.init(rngs=jax.random.PRNGKey(0), method="init_pass")[
    "params"
]

pretrained_params = {"cat_codec": cat_codec_params}
init_fn_dict = {}
model = MetaLearner(
    codec_in="struct_codec",
    model_dict=modules_dict,
    pretrained_params_dict=pretrained_params,
    init_fn_dict=init_fn_dict,
)

# ---------------TRAINING CONFIG------------------------
optimizer_dp = OptimizerConfig(
    optim="adam",
    is_dp=True,
    clipping_norm=1e-3,
    noise_multiplier=1.0,
    learning_rate=1e-2,
    gradient_accumulation_steps=GRADS_ACCUM,
    weight_decay=False,
    load_state=False,
    state_dir="",  # no lr decay
)
optimizer_no_dp = OptimizerConfig(
    optim="adam",
    is_dp=False,
    noise_multiplier=0.0,  # not used in training because of flag above
    clipping_norm=1e9,  # not used in training because of flag above
    learning_rate=1e-3,
    gradient_accumulation_steps=GRADS_ACCUM,
    weight_decay=False,
    load_state=False,
    state_dir="",  # no lr decay
)

save_dir = os.path.join(str(Path(__file__).parent), "shared_codec_example_results")
output_dir = (
    os.path.join(save_dir, "dp_training")
    if TRAIN_DP
    else os.path.join(
        save_dir,
        "standard_training",
    )
)

dp_check_point_config = CheckpointConfig(
    output_dir=output_dir,
    overwrite_output_dir=True,
    logging_steps=GRADS_ACCUM,
    save_every_steps=SAVE_EVERY,  # TODO: set real params
    tensorboard_dir=output_dir + "/tensorboard",
)

standard_chkpnt_config = CheckpointConfig(
    output_dir=output_dir,
    overwrite_output_dir=True,
    logging_steps=GRADS_ACCUM,
    save_every_steps=SAVE_EVERY,  # TODO: set real params
    tensorboard_dir=output_dir + "/tensorboard",
)

training_config = (
    TrainingConfig(
        random_seed=0,
        optimizer_config=optimizer_dp,
        check_point_config=dp_check_point_config,
        batch_size=TRAIN_BATCH_SIZE,
        params_dtype="float32",
        num_train_steps=TRAIN_STEPS,
        eval_every_step=EVAL_EVERY,
    )
    if TRAIN_DP
    else TrainingConfig(
        random_seed=0,
        optimizer_config=optimizer_no_dp,
        check_point_config=standard_chkpnt_config,
        batch_size=TRAIN_BATCH_SIZE,
        params_dtype="float32",
        num_train_steps=TRAIN_STEPS,
        eval_every_step=EVAL_EVERY,
    )
)

split = int(0.9 * N_obs)

train_set, test_set = jax.tree_map(
    lambda x: x[:split],
    dataset,
), jax.tree_map(
    lambda x: x[split:],
    dataset,
)


def jax_iterator(dataset):
    ds_len = len(dataset[0])
    while True:
        for i in range(int(ds_len / TRAIN_BATCH_SIZE)):
            yield jax.tree_map(
                lambda x: x[i * TRAIN_BATCH_SIZE : (i + 1) * TRAIN_BATCH_SIZE],
                dataset,
            )


def test_iterator():
    eval_len = len(test_set[0])
    for i in range(int(eval_len / TEST_BATCH_SIZE)):
        yield jax.tree_map(
            lambda x: x[i * TEST_BATCH_SIZE : (i + 1) * TEST_BATCH_SIZE],
            test_set,
        )


### Customize the training and train:
rng = jax.random.PRNGKey(0)
model_params = jax.jit(model.init)(rng=rng)
options = CheckpointManagerOptions(max_to_keep=3, keep_period=2)
mngr = CheckpointManager(
    training_config.check_point_config.output_dir,
    PyTreeCheckpointer(),
    options=options,
)

if RESTORE:
    shift_step = SHIFT_STEP
    restored = mngr.restore(shift_step)
    model_params = restored["model_params"]
    dp_state = None  # restored['standard_state']
    standard_state = None  # restored['dp_state']

else:
    dp_state = None
    standard_state = None
    shift_step = 0

if TRAIN:
    model_params, _, _ = train(
        model=model,
        dataset=jax_iterator(train_set),
        eval_set=test_iterator,
        model_params=model_params,
        training_config=training_config,
        standard_state=standard_state,
        dp_state=dp_state,
        shift_step=shift_step,
    )

s = model.sample(model_params, rng=jax.random.PRNGKey(0), size=10)


print(s)
