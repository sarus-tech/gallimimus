import jax
import jax.numpy as jnp

from codec.categorical_codec import CategoricalCodec
from codec.list_codec import ListCodec
from codec.struct_codec import StructCodec
from model import BatchMetaLearner

from training import TrainingHyperparameters, train

from jax import config

config.update("jax_debug_nans", True)

embed_dim = 8

### Create the model
cat_codec = CategoricalCodec(
    name="cat0",
    embed_dim=embed_dim,
    vocab_size=20,
)


l_codec = ListCodec(
    embed_dim=embed_dim,
    subcodec=cat_codec,
    max_len=10,
    buffer_size=5,
    n_heads=4,
    n_blocks=1,
)

struct_codec = StructCodec(
    embed_dim=embed_dim, n_heads=4, n_blocks=1, subcodecs=[cat_codec, cat_codec]
)


l_codec2 = ListCodec(
    embed_dim=embed_dim,
    subcodec=l_codec,
    max_len=8,
    buffer_size=4,
    n_heads=4,
    n_blocks=1,
)

model = BatchMetaLearner(
    codec=l_codec2,
)

### Train the model
rng = jax.random.PRNGKey(0)

ds = [model.example for i in range(500)]

params = model.init(rng=rng)

hyperparams = TrainingHyperparameters(
    num_epochs=10,
    batch_size=100,
    learning_rate=1e-1,
    dp=True,
    noise_multiplier=0.3,
    l2_norm_clip=1.,
)

trained_params = train(model=model, params=params, hyperparams=hyperparams, dataset=ds)

s = model.sample(trained_params, rng=jax.random.PRNGKey(0), size=5)

print(s)
