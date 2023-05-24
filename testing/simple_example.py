import jax
import jax.numpy as jnp

from gallimimus.codec import CategoricalCodec, ListCodec, StructCodec
from gallimimus.model import BatchMetaLearner

from gallimimus.training import TrainingHyperparameters, train
import flax.linen as nn
from jax import config

config.update("jax_debug_nans", True)

embed_dim = 8
N = 10000

# let's create a dataset where each observation is
# (mu ~ U([0,100[) , [mu + x_i where x_i ~ N(0,10) for i in range(l)] with l ~ P(lambda) in list_length_is_gt_lambda,

rng = jax.random.PRNGKey(0)

rng1, rng2, rng = jax.random.split(rng, 3)
lam = 10
max_len = 2*lam

buffer_size = lam
max_mu = 100

mus = jax.random.categorical(key=rng1, logits=jnp.ones((max_mu,)), shape=(N,)).astype(int)

lens = jax.random.poisson(key=rng2,lam=lam, shape=(N,), dtype=int ).clip(max=max_len-1)

rngs = jax.random.split(rng, N * max_len)

mus_noised = mus[:, None] + jax.random.normal(key=rng, shape=(N, buffer_size)).astype(int)
mus_noised = mus_noised.clip(max=max_mu-1)

dataset = [(mus[i], (lens[i], mus_noised[i]), int(lens[i] >= lam)) for i in range(N)]

### Create the model
mus_codec = CategoricalCodec(
    embed_dim=embed_dim,
    vocab_size=max_mu,
)

cat_len_codec = CategoricalCodec(
    embed_dim=embed_dim,
    vocab_size=max_len,
)

l_codec = ListCodec(
    embed_dim=embed_dim,
    subcodec_in = mus_codec,
    max_len=max_len*2,
    buffer_size=lam,
    n_heads=4,
    n_blocks=1,
)

is_long_codec = CategoricalCodec(
    embed_dim=embed_dim,
    vocab_size = 2,
)

struct_codec = StructCodec(
    embed_dim=embed_dim,
    n_heads=4,
    n_blocks=1,
    subcodecs_in=[mus_codec, l_codec, is_long_codec],
)




model = BatchMetaLearner(
    codec_in=struct_codec,
)



### Train the model
rng = jax.random.PRNGKey(0)
N = 5000




params = model.init(rng=rng)



hyperparams = TrainingHyperparameters(
    num_epochs=100,
    batch_size=100,
    learning_rate=1e-1,
    dp=False,
    noise_multiplier=0.3,
    l2_norm_clip=1.0,
)

trained_params = train(model=model, params=params, hyperparams=hyperparams, dataset=dataset)

s = model.sample(trained_params, rng=jax.random.PRNGKey(0), size=5)

print(s)
