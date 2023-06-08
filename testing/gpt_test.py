
import jax
import jax.random
import jax.numpy as jnp
import numpy as np
import optax
from faker.providers.person.en import Provider
from transformers import AutoTokenizer, FlaxGPT2Model, AutoConfig
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2Module

from gallimimus import MetaLearner, TrainingHyperparameters, train
from gallimimus.codec.text_codec import TextCodec

# config = AutoConfig.from_pretrained("gpt2")
# 
# # model = FlaxGPT2LMHeadModel.from_config(config)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

#jax.config.update("jax_debug_nans", True)

def make_names(size_train, size_eval):
    first_names = list(set(Provider.first_names))
    last_names = list(set(Provider.last_names))

    first_names_r = np.random.choice(first_names, size=size_train + size_eval)
    last_names_r = np.random.choice(last_names, size=size_train + size_eval)

    names = [ f"{first_name} {last_name}" for first_name, last_name in zip(first_names_r, last_names_r)]

    tokenized_list_raw = tokenizer(names, padding='longest').data
    
    position_ids = jnp.arange(len(tokenized_list_raw["input_ids"][0]))
    tokenized_list = [  {"attention_mask": jnp.array(am), "input_ids": jnp.array(ii), "position_ids":position_ids}  for am, ii in zip(tokenized_list_raw["attention_mask"], tokenized_list_raw["input_ids"])]
    
    return tokenized_list[:size_train], tokenized_list[size_train:]


ds, ds_eval = make_names(100, 100)


######
config_gpt2, model_kwargs = FlaxGPT2Model.config_class.from_pretrained(
    "distilgpt2",
    return_unused_kwargs=True,
    force_download=False,
    resume_download=False,
    local_files_only=False,
    _from_auto=False,
)
model_kwargs["dtype"] = jnp.float32

gpt2_model = FlaxGPT2Module(config_gpt2, **model_kwargs)

#######


gpt2_model_config = AutoConfig.from_pretrained("distilgpt2")
gptmodel_params = FlaxGPT2Model(gpt2_model_config).params
gptmodel = FlaxGPT2Module(gpt2_model_config)

text_codec = TextCodec(
    embed_dim = 16,
    n_tokens = 10,
    max_length = 100,
    model_name = "distilgpt2"
)

model_dict = {
    "text_codec": text_codec,
    "distilgpt2": gptmodel
}

params_dict = {
    "distilgpt2": gptmodel_params
}
model = MetaLearner(
    codec_in="text_codec",
    model_dict=model_dict,
    pretrained_params_dict=params_dict,
)


rng = jax.random.PRNGKey(0)
params = jax.jit(model.init)(rng=rng)

hyperparams = TrainingHyperparameters(
    num_epochs=100,
    batch_size=1,

    dp=False,
    noise_multiplier=0.3,
    l2_norm_clip=1.0,
)


optimizer = optax.sgd(
    learning_rate=1e-1,
)
trained_params = train(
    model=model,
    params=params,
    optimizer=optimizer,
    hyperparams=hyperparams,
    dataset=ds,
    eval_dataset=ds_eval,
)

s = model.sample(trained_params, rng=jax.random.PRNGKey(0), size=10)

print(s)
