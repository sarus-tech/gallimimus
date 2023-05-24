import jax
import numpy as np

from gallimimus.codec import CategoricalCodec, ListCodec, StructCodec
from gallimimus.model import BatchMetaLearner

from gallimimus.training import TrainingHyperparameters, train
import flax.linen as nn
from jax import config

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
    subcodec_in = cat_codec,
    max_len=max_len,
    buffer_size=buffer_size,
    n_heads=4,
    n_blocks=1,
)

struct_codec = StructCodec(
    embed_dim=embed_dim,
    n_heads=4,
    n_blocks=1,
    subcodecs_in=[cat_codec, cat_len_codec, l_codec],
)


# struct_codec = StructCodec(
#     embed_dim=embed_dim,
#     n_heads=4,
#     n_blocks=1,
#     subcodecs_in=[cat_codec, l_codec],
# )

model = BatchMetaLearner(
    codec_in=struct_codec,
)



### Train the model
rng = jax.random.PRNGKey(0)
N = 5000

rng1, rng2, rng = jax.random.split(rng, 3)
maxs = jax.random.randint(key=rng1, shape=(N,), minval=0, maxval=20)
lens = jax.random.poisson(key=rng2, lam=20, shape=(N,)).clip(0, max_len)

rngs = jax.random.split(rng, N)
b = []
ds = [
    (
        maxs[i],
        lens[i],
        (
            lens[i],
            jax.random.randint(
                rngs[i],
                [
                    buffer_size,
                ],
                0,
                maxs[i],
            ),
        ),
    )
    for i in range(N)
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
