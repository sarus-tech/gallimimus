"""pretrained distilgpt2 trained with LoRA

taken from the notebook https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb
listed at https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#resources"""
import os
import pickle
import time

import datasets
import jax
import numpy as np
import optax
import jax.numpy as jnp
import pandas as pd
from flax.training import train_state
from tqdm import tqdm
import jax.random
from transformers import FlaxAutoModel, AutoTokenizer
from datasets import load_dataset

from faker.providers.person.en import Provider

from lora_flax import LoRA

import logging
logging.basicConfig(filename='./myapp.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

### import the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

model = FlaxAutoModel.from_pretrained("distilgpt2")

apply_fn = lambda params, input, **kwargs: model(
    **input, params=params["params"], **kwargs
)


### definition of the different datasets:
def make_names(size_train, size_eval):
    first_names = list(set(Provider.first_names))
    last_names = list(set(Provider.last_names))

    first_names_r = np.random.choice(first_names, size=size_train + size_eval)
    last_names_r = np.random.choice(last_names, size=size_train + size_eval)

    names = [ f"{first_name} {last_name}" for first_name, last_name in zip(first_names_r, last_names_r)]

    raw_ds = {"train": names[:size_train], "validation": names[size_train:]}
    tokenized_ds = {k: tokenizer(v, padding='longest') for k,v in raw_ds.items()}

    for k, v in tokenized_ds.items():
        v["labels"] = v["input_ids"].copy()

    tokenized_ds = {k: datasets.Dataset.from_dict(v) for k,v in tokenized_ds.items()}
    return tokenized_ds

def make_huggingface_ds(name, max_seq_length, text_name="text"):
    raw_dataset = load_dataset(*name)
    raw_dataset["train"] = load_dataset(
        *name, split="train[5%:]"
    )
    raw_dataset["validation"] = load_dataset(
        *name, split="train[:5%]"
    )

    # TODO these cells should be commented out to run on full dataset
    raw_dataset["train"] = raw_dataset["train"].select(range(200))
    raw_dataset["validation"] = raw_dataset["validation"].select(range(20))


    def tokenize_function(examples):
        out = tokenizer(examples[text_name], padding="max_length", max_length=max_seq_length, truncation=True)
        return out


    tokenized_datasets1 = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=raw_dataset["train"].column_names,
    )


    def group_texts(examples):
        # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # total_length = len(concatenated_examples[list(examples.keys())[0]])
        # total_length = (total_length // max_seq_length) * max_seq_length
        # result = {
        #     k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        #     for k, t in concatenated_examples.items()
        # }
        result = {k: examples[k] for k in examples.keys()}
        result["labels"] = result["input_ids"].copy()
        return result


    tokenized_datasets = tokenized_datasets1.map(group_texts, batched=True, num_proc=8)

    return tokenized_datasets


def make_oscar():
    language = "is"
    max_seq_length = 512
    name = ("oscar", f"unshuffled_deduplicated_{language}")

    return make_huggingface_ds(
        name=name,
        max_seq_length=max_seq_length,
    )

def make_reviews():
    name = ("flxclxc/encoded_drug_reviews",)
    dataset = make_huggingface_ds(
        name=name,
        max_seq_length=128,
        text_name="review"
    )
    return dataset


def count_params(params):
    param_count = sum(x.size for x in jax.tree_leaves(params))
    return param_count

### main training function:

def do_lora_training(filter_fn, lora_rank, save_folder, tokenized_datasets):
    os.makedirs(save_folder, exist_ok=True)
    print(f"--- saving to {save_folder} ---")
    try:
        t0 = time.perf_counter()

        lora_bert = LoRA(
            target_apply_fn=apply_fn,
            pretrained_params=model.params,
            filter_fn=filter_fn,
            r=lora_rank,
        )

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

        eval_file = os.path.join(save_folder, "eval_metric.csv")
        train_file = os.path.join(save_folder, "train_metric.csv")

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
                t1 = time.perf_counter() - t0
                train_metrics_save = {"time": t1, "loss": train_metric["loss"], "epoch": epoch}
                train_metrics_save = pd.DataFrame.from_records([train_metrics_save])
                train_metrics_save.to_csv(train_file, index=False, mode='a', header=not os.path.exists(train_file))
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

                t2 = time.perf_counter() - t0
                eval_metrics_save = {"time": t2, "loss": eval_metrics["loss"], "perplexity": eval_metrics["perplexity"]}
                eval_metrics_save = pd.DataFrame.from_records([eval_metrics_save])
                eval_metrics_save.to_csv(eval_file, index=False, mode='a', header=not os.path.exists(eval_file))



    except Exception as err:
        raise
        logger.error(err)



if __name__ == "__main__":
    tokenized_datasets = make_names(size_train=10000, size_eval=100)

    filter_fn1 = lambda param_name, params: "kernel" in param_name and "attn" in param_name

    for r in [2, 4, 8]:
        do_lora_training(
            filter_fn=filter_fn1,
            lora_rank = r,
            save_folder=f"./exp1_{r}",
            tokenized_datasets = tokenized_datasets,
        )

    ### exp 2
    tokenized_datasets = make_reviews()

    r = 4

    do_lora_training(
        filter_fn=filter_fn1,
        lora_rank = r,
        save_folder="./exp2",
        tokenized_datasets=tokenized_datasets,
    )

    ### exp 3
    tokenized_datasets = make_oscar()

    r = 2

    do_lora_training(
        filter_fn=filter_fn1,
        lora_rank=r,
        save_folder="./exp3"
    )