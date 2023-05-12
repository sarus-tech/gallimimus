"""
LoRA example with linear regression
"""

import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import flax

from lora_flax import LoRA


########################

# Set problem dimensions.
n_samples = 20
x_dim = 10
y_dim = 5

# Generate random ground truth W and b.
key = random.PRNGKey(0)
k1, k2 = random.split(key)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim,))
# Store the parameters in a FrozenDict pytree.
true_params = flax.core.freeze({"params": {"bias": b, "kernel": W}})

# Generate samples with additional noise.
key_sample, key_noise = random.split(k1)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = (
    jnp.dot(x_samples, W) + b + 0.1 * random.normal(key_noise, (n_samples, y_dim))
)
print("x shape:", x_samples.shape, "; y shape:", y_samples.shape)


##################


model_pre = nn.Dense(features=5)
rng = random.PRNGKey(0)

test_input = random.normal(rng, (10,))

pretrained_params = model_pre.init({"params": rng}, test_input)["params"]
pretrained_params = flax.core.freeze(
    {"bias": b, "kernel": jnp.zeros_like(pretrained_params["kernel"])}
)


model = LoRA(
    target_apply_fn=model_pre.apply,
    pretrained_params=pretrained_params,
    filter_fn=lambda param_name, param: len(param.shape) == 2,
    r=5,
)

lora_params = model.init({"params": rng}, test_input)

model.apply(lora_params, test_input)


def mse(params, x_batched, y_batched):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = model.apply(params, x)
        return jnp.inner(y - pred, y - pred) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


learning_rate = 1e-1  # Gradient step size.

loss_grad_fn = jax.value_and_grad(mse)


@jax.jit
def update_params(params, learning_rate, grads):
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params


params = lora_params
for i in range(201):
    # Perform one gradient update.
    loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
    params = update_params(params, learning_rate, grads)
    if i % 10 == 0:
        a, b = params["params"]["lora"]["kernel"]
        print(
            f"Loss step {i:<5}: {loss_val:.4f}, norm = {jnp.linalg.norm(W - a@b):.4f}"
        )
