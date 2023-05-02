import jax
import numpy as np

from minimal_synthetic_data.codec.categorical_codec import CategoricalCodec
from minimal_synthetic_data.codec.list_codec import ListCodec
from minimal_synthetic_data.codec.struct_codec import StructCodec
from minimal_synthetic_data.model import BatchMetaLearner

from minimal_synthetic_data.training import TrainingHyperparameters, train

from jax import config

config.update("jax_debug_nans", True)

embed_dim = 8

### Create the model
cat_codec = CategoricalCodec(
    embed_dim=embed_dim,
    vocab_size=20,
)

max_len = 50
buffer_size = 20
cat_len_codec = CategoricalCodec(
    embed_dim=embed_dim,
    vocab_size=max_len,
)

l_codec = ListCodec(
    embed_dim=embed_dim,
    subcodec=cat_codec,
    max_len=max_len,
    buffer_size=buffer_size,
    n_heads=4,
    n_blocks=1,
)

struct_codec = StructCodec(
    embed_dim=embed_dim,
    n_heads=4,
    n_blocks=1,
    subcodecs=[cat_codec, cat_len_codec, l_codec],
)

model = BatchMetaLearner(
    codec=struct_codec,
)

### Train the model
rng = jax.random.PRNGKey(0)
N = 5000

rng1, rng2, rng = jax.random.split(rng, 3)
maxs = jax.random.randint(key=rng1, shape=(N,), minval=0, maxval=20)
lens = jax.random.poisson(key=rng2, lam=20, shape=(N,)).clip(0, max_len)

rngs = jax.random.split(rng, N)
b = []
ds = [(
    maxs[i], lens[i], (
        lens[i],
        jax.random.randint(rngs[i],[buffer_size,],0,maxs[i]),
        ),
    ) for i in range(N)
]


params = model.init(rng=rng)

hyperparams = TrainingHyperparameters(
    num_epochs=10,
    batch_size=100,
    learning_rate=1e-1,
    dp=False,
    noise_multiplier=0.3,
    l2_norm_clip=1.0,
)

trained_params = train(model=model, params=params, hyperparams=hyperparams, dataset=ds)

s = model.sample(trained_params, rng=jax.random.PRNGKey(0), size=5)

print(s)
