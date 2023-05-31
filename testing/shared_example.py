import jax
import jax.numpy as jnp
import optax

from gallimimus import MetaLearner, TrainingHyperparameters, train
from gallimimus.codec import (
    CategoricalCodec,
    ListCodec,
    StructCodec,
    SharedCodec,
    LoraCodec,
)

from jax import config

config.update("jax_debug_nans", True)

embed_dim = 16
N = 100

max_cat = 10

dataset = [(jnp.array(i % max_cat), jnp.array(i**2 % max_cat)) for i in range(N)]


### Create the model
cat_codec = CategoricalCodec(
    embed_dim=embed_dim,
    vocab_size=max_cat,
)

shared_cat_codec = SharedCodec(embed_dim=embed_dim, shared_module_name="cat_codec")

model_dict = {"cat_codec": cat_codec}

### with split lora params

lora_codec = LoraCodec(
    embed_dim=embed_dim,
    subcodec_in=shared_cat_codec,
    lora_module_name="cat_codec",
    filter_fn=lambda path, arr: len(arr.shape) == 2,
    r=2,
)

struct_codec = StructCodec(
    embed_dim=embed_dim,
    n_heads=8,
    n_blocks=2,
    subcodecs_in=[lora_codec, lora_codec],
)

### with shared lora params

struct_codec2 = StructCodec(
    embed_dim=embed_dim,
    n_heads=8,
    n_blocks=2,
    subcodecs_in=[shared_cat_codec, shared_cat_codec],
)

lora_codec2 = LoraCodec(
    embed_dim=embed_dim,
    subcodec_in=struct_codec2,
    lora_module_name="cat_codec",
    filter_fn=lambda path, arr: len(arr.shape) == 2,
    r=2,
)

####


model = MetaLearner(
    codec_in=lora_codec2,
    model_dict=model_dict,
    params_dict={},
)


### Train the model
rng = jax.random.PRNGKey(0)


params = model.init(rng=rng)


hyperparams = TrainingHyperparameters(
    num_epochs=10,
    batch_size=10,
    dp=False,
    noise_multiplier=0.3,
    l2_norm_clip=1.0,
)

split = int(0.95 * N)

optimizer = optax.adamaxw(
    learning_rate=1e-1,
)
trained_params = train(
    model=model,
    params=params,
    optimizer=optimizer,
    hyperparams=hyperparams,
    dataset=dataset[:split],
    eval_dataset=dataset[split:],
)

s = model.sample(trained_params, rng=jax.random.PRNGKey(0), size=1)

print(s)
