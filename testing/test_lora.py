"""pretrained distilgpt2 trained with LoRA

taken from the notebook https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb
listed at https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#resources"""
import jax
import optax
import jax.numpy as jnp
from flax.training import train_state
from tqdm import tqdm
import jax.random
from transformers import FlaxAutoModel, AutoTokenizer
from datasets import load_dataset

from lora_flax import LoRA

### import the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = FlaxAutoModel.from_pretrained("distilgpt2")


### create the LoRA model
apply_fn = lambda params, input, **kwargs: model(
    **input, params=params["params"], **kwargs
)

filter_fn = lambda param_name, params: param_name == (
    "h",
    "0",
    "attn",
    "c_attn",
    "kernel",
)

lora_bert = LoRA(
    target_apply_fn=apply_fn,
    pretrained_params=model.params,
    filter_fn=filter_fn,
    r=4,
)


### download and tokenize the dataset
language = "is"
max_seq_length = 512

raw_dataset = load_dataset("oscar", f"unshuffled_deduplicated_{language}")
raw_dataset["train"] = load_dataset(
    "oscar", f"unshuffled_deduplicated_{language}", split="train[5%:]"
)
raw_dataset["validation"] = load_dataset(
    "oscar", f"unshuffled_deduplicated_{language}", split="train[:5%]"
)

# these cells should be commented out to run on full dataset
raw_dataset["train"] = raw_dataset["train"].select(range(200))
raw_dataset["validation"] = raw_dataset["validation"].select(range(20))


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets1 = raw_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=raw_dataset["train"].column_names,
)


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_datasets = tokenized_datasets1.map(group_texts, batched=True, num_proc=4)

### train the LoRA model with the dataset
batch_size = 16
num_epochs = 10
training_seed = 0
learning_rate = 3e-4

num_train_steps = len(tokenized_datasets["train"]) // batch_size * num_epochs

linear_decay_lr_schedule_fn = optax.linear_schedule(
    init_value=learning_rate, end_value=0, transition_steps=num_train_steps
)
adamw = optax.adamw(
    learning_rate=linear_decay_lr_schedule_fn,
    b1=0.9,
    b2=0.98,
    eps=1e-8,
    weight_decay=0.01,
)


def data_loader(rng, dataset, batch_size, shuffle=False):
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}

        yield batch


init_rng = jax.random.PRNGKey(0)

init_loader = data_loader(
    init_rng, tokenized_datasets["train"], batch_size, shuffle=True
)

init_batch = next(init_loader)
labels = init_batch.pop("labels")
lora_param = lora_bert.init(jax.random.PRNGKey(0), init_batch)

state = train_state.TrainState.create(
    apply_fn=lora_bert.apply, params=lora_param, tx=adamw
)


@jax.jit
def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        labels = batch.pop("labels")
        logits = state.apply_fn(
            variables=params, input=batch, dropout_rng=dropout_rng, train=True
        )[0]

        loss = optax.softmax_cross_entropy(
            logits[..., :-1, :], jax.nn.one_hot(labels[..., 1:], logits.shape[-1])
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grad)

    metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}

    return new_state, metrics, new_dropout_rng


def eval_step(params, batch):
    labels = batch.pop("labels")

    logits = lora_bert.apply(variables=params, input=batch, train=False)[0]

    loss = optax.softmax_cross_entropy(
        logits[..., :-1, :], jax.nn.one_hot(labels[..., 1:], logits.shape[-1])
    ).mean()

    # summarize metrics
    metrics = {"loss": loss, "perplexity": jnp.exp(loss)}
    return metrics


rng = jax.random.PRNGKey(training_seed)
dropout_rngs = rng

for epoch in tqdm(range(1, num_epochs + 1), desc=f"Epoch ...", position=0, leave=True):
    rng, input_rng = jax.random.split(rng)

    # -- Train --
    train_loader = data_loader(
        input_rng, tokenized_datasets["train"], batch_size, shuffle=True
    )
    with tqdm(
        total=len(tokenized_datasets["train"]) // batch_size,
        desc="Training...",
        leave=False,
    ) as progress_bar_train:
        for model_inputs in train_loader:
            # Model forward
            state, train_metric, dropout_rngs = train_step(
                state, model_inputs, dropout_rngs
            )

            progress_bar_train.update(1)

        progress_bar_train.write(
            f"Train... ({epoch}/{num_epochs} | Loss: {round(train_metric['loss'].mean(), 3)}, Learning Rate: {round(train_metric['learning_rate'].mean(), 6)})"
        )

    # -- Eval --
    eval_loader = data_loader(input_rng, tokenized_datasets["validation"], batch_size)
    eval_metrics = []

    with tqdm(
        total=len(tokenized_datasets["validation"]) // batch_size,
        desc="Evaluation...",
        leave=False,
    ) as progress_bar_eval:
        for model_inputs in eval_loader:
            # Model forward
            eval_metric = eval_step(state.params, model_inputs)
            eval_metrics.append(eval_metric)

            progress_bar_eval.update(1)

        eval_metrics = jax.tree_map(
            lambda *leaves: jnp.mean(jnp.array(leaves)), *eval_metrics
        )
        progress_bar_eval.write(
            f"Eval... ({epoch}/{num_epochs} | Loss: {eval_metrics['loss']} | Perplexity: {eval_metrics['perplexity']})"
        )
