import jax.random
import jax.numpy as jnp
import numpy as np
from transformers import FlaxGPT2Model, AutoConfig
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2Module

from gallimimus import MetaLearner
from gallimimus.codec import LoraCodec
from gallimimus.codec.text_codec import TextCodec
from gallimimus.training.configs import (
    TrainingConfig,
    OptimizerConfig,
    CheckpointConfig,
)
from gallimimus.training.training import train
import logging
import os
from pathlib import Path
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)
from transformers import AutoTokenizer

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

EMBED_DIM = 16
N_TOKENS = 10
MAX_LENGTH = 100
TEXT_MODEL = "distilgpt2"
LORA_RANK = 4
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = TRAIN_BATCH_SIZE
TRAIN_STEPS = 15000
GRADS_ACCUM = 1
TRAIN_DP = False
DS_LENGTH = 10000
EVAL_LENGTH = 500
EVAL_EVERY = 15 * GRADS_ACCUM
SAVE_EVERY = 50 * GRADS_ACCUM
TRAIN = False
RESTORE = True
SHIFT_STEP = 5500
DO_LORA = False

### Build the model
"""
Building the GPT2 model

Warning! Monkeypatched to expose internal methods
"""
gpt2_model_config = AutoConfig.from_pretrained(TEXT_MODEL)
gptmodel_params = FlaxGPT2Model(gpt2_model_config).params
gptmodel = FlaxGPT2Module(gpt2_model_config)


def make_fn(method_name):
    def fn(self, *args, **kwargs):
        return getattr(self, method_name)(*args, **kwargs)

    return fn


for method_name in ["wpe", "wte", "h", "ln_f", "wpe_attend", "wte_attend"]:
    setattr(FlaxGPT2Module, f"_{method_name}", make_fn(method_name))


def make_fn_attend(method_name):
    def fn(self, *args, **kwargs):
        return getattr(self, method_name).attend(*args, **kwargs)

    return fn


for method_name in ["wpe", "wte"]:
    setattr(FlaxGPT2Module, f"_{method_name}_attend", make_fn_attend(method_name))

text_codec = TextCodec(
    embed_dim=EMBED_DIM,
    n_tokens=N_TOKENS,
    max_length=MAX_LENGTH,
    model_name="distilgpt2",
)

lora_codec = LoraCodec(
    embed_dim=EMBED_DIM,
    subcodec_in="text_codec",
    lora_module_name=TEXT_MODEL,
    filter_fn=lambda path, arr: "kernel" in path,
    r=LORA_RANK,
)


if DO_LORA:
    modules_dict = {
    "text_codec": text_codec,
    "lora_codec": lora_codec,
    "distilgpt2": gptmodel,
}


    pretrained_params = {"distilgpt2": gptmodel_params}
    init_fn_dict = {}
    model = MetaLearner(
        codec_in="lora_codec",
        model_dict=modules_dict,
        pretrained_params_dict=pretrained_params,
        init_fn_dict=init_fn_dict
)
else:
    modules_dict = {
    "text_codec": text_codec,
    "distilgpt2": gptmodel,
}

    pretrained_params = {}
    init_fn_dict = {"distilgpt2": lambda rng: gptmodel_params}
    model = MetaLearner(
        codec_in="text_codec",
        model_dict=modules_dict,
        pretrained_params_dict=pretrained_params,
        init_fn_dict=init_fn_dict
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

save_dir = os.path.join(str(Path(__file__).parent), "gpt2_exp", "reviews")
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


#### make dataset

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token


from faker.providers.person.en import Provider


def make_names(size_train, size_eval):
    first_names = list(set(Provider.first_names))
    last_names = list(set(Provider.last_names))

    full_size = size_train + size_eval
    first_names_r = np.random.choice(first_names, size=full_size)
    last_names_r = np.random.choice(last_names, size=full_size)

    names = [
        f"{first_name} {last_name}"
        for first_name, last_name in zip(first_names_r, last_names_r)
    ]

    tokenized_data = tokenizer(names, padding="longest").data
    tokenized_data = {k: jnp.array(v) for k, v in tokenized_data.items()}

    tokenized_data["input_ids"] = jnp.concatenate(
        [jnp.full((full_size, 1), tokenizer.bos_token_id), tokenized_data["input_ids"]],
        axis=-1,
    )
    tokenized_data["attention_mask"] = jnp.concatenate(
        [jnp.ones((full_size, 2)), tokenized_data["attention_mask"][:, :-1]], axis=-1
    )
    position_ids = jnp.cumsum(tokenized_data["attention_mask"], axis=-1)
    tokenized_data["position_ids"] = position_ids

    dataset = jax.tree_util.tree_map(lambda arr: arr[:size_train], tokenized_data)
    test_set = jax.tree_util.tree_map(lambda arr: arr[size_train:], tokenized_data)

    return dataset, test_set


train_set, test_set = make_names(DS_LENGTH, EVAL_LENGTH)


def jax_iterator(dataset):
    while True:
        for i in range(int(DS_LENGTH / TRAIN_BATCH_SIZE)):
            yield jax.tree_map(
                lambda x: x[i * TRAIN_BATCH_SIZE : (i + 1) * TRAIN_BATCH_SIZE], dataset
            )


def test_iterator():
    for i in range(int(EVAL_LENGTH / TEST_BATCH_SIZE)):
        yield jax.tree_map(
            lambda x: x[i * TEST_BATCH_SIZE : (i + 1) * TEST_BATCH_SIZE], test_set
        )


### Customize the training and train:
rng = jax.random.PRNGKey(0)
model_params = jax.jit(model.init)(rng=rng)
options = CheckpointManagerOptions(max_to_keep=3, keep_period=2)
mngr = CheckpointManager(
    training_config.check_point_config.output_dir, PyTreeCheckpointer(), options=options
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


text = tokenizer.batch_decode(s)

print(text)
