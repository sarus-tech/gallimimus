import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn

import numpy as np

from gallimimus import MetaLearner, TrainingHyperparameters, train
from gallimimus.codec import CategoricalCodec, ListCodec, StructCodec

from transformers import AutoTokenizer, FlaxAutoModel
from faker.providers.person.en import Provider

from jax import config

config.update("jax_debug_nans", True)

embed_dim = 32
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

model = FlaxAutoModel.from_pretrained("distilgpt2")

apply_fn = lambda params, input, **kwargs: model(
    **input, params=params["params"], **kwargs
)

embed_dim = 8
N = 10000

def make_names(size_train, size_eval):
    first_names = list(set(Provider.first_names))
    last_names = list(set(Provider.last_names))

    first_names_r = np.random.choice(first_names, size=size_train + size_eval)
    last_names_r = np.random.choice(last_names, size=size_train + size_eval)

    names = [ f"{first_name} {last_name}" for first_name, last_name in zip(first_names_r, last_names_r)]

    tokenized_list = [ tokenizer(v, padding='longest') for v in names]

    return tokenized_list


# let's create a dataset where each observation is
# (mu ~ U([0,100[) , [mu + x_i where x_i ~ N(0,10) for i in range(l)] with l ~ P(lambda) in list_length_is_gt_lambda,

rng = jax.random.PRNGKey(0)

rng1, rng2, rng = jax.random.split(rng, 3)
lam = 10
max_len = 2 * lam

buffer_size = lam
max_mu = 100

mus = jax.random.categorical(key=rng1, logits=jnp.ones((max_mu,)), shape=(N,)).astype(
    int
)

lens = jax.random.poisson(key=rng2, lam=lam, shape=(N,), dtype=int).clip(
    max=max_len - 1
)

rngs = jax.random.split(rng, N * max_len)

mus_noised = mus[:, None] + jax.random.normal(key=rng, shape=(N, buffer_size)).astype(
    int
)
mus_noised = mus_noised.clip(max=max_mu - 1)

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
    subcodec_in=mus_codec,
    max_len=max_len * 2,
    buffer_size=lam,
    n_heads=8,
    n_blocks=2,
)

is_long_codec = CategoricalCodec(
    embed_dim=embed_dim,
    vocab_size=2,
)

struct_codec = StructCodec(
    embed_dim=embed_dim,
    n_heads=8,
    n_blocks=2,
    subcodecs_in=[mus_codec, l_codec, is_long_codec],
)

MLP = nn.Dense(10)
mlp_params = MLP.init(rngs=rng, inputs=jnp.zeros((5)))

model = MetaLearner(codec_in=struct_codec, model_dict={"mlp": (MLP.apply, mlp_params)})


### Train the model
rng = jax.random.PRNGKey(0)


params = model.init(rng=rng)


hyperparams = TrainingHyperparameters(
    num_epochs=100,
    batch_size=1000,

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
