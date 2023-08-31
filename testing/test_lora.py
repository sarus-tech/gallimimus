"""Pretrained distilgpt2 trained with LoRA.

taken from the notebook
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb
listed at https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#resources
"""
import logging
import os
import time
import typing
from dataclasses import dataclass

import datasets
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import pandas as pd
from datasets import load_dataset
from faker.providers.person.en import Provider
from flax.training import train_state
from tqdm import tqdm
from transformers import AutoTokenizer, FlaxAutoModel

from lora_flax import LoRA

logging.basicConfig(
    filename="./myapp.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

### import the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

model = FlaxAutoModel.from_pretrained("distilgpt2")


def apply_fn(params, input, **kwargs):
    return model(**input, params=params["params"], **kwargs)


# wrap the language model, so that it uses the usual Flax interface:
@dataclass
class WrapperModel:
    model: typing.Any

    def apply(self, params, input, rngs={}, method=None, **kwargs):
        return self.model(**input, params=params["params"], **kwargs)


wrapped_model = WrapperModel(model)


### definition of the different datasets:
def make_names(size_train, size_eval):
    first_names = list(set(Provider.first_names))
    last_names = list(set(Provider.last_names))

    first_names_r = np.random.choice(first_names, size=size_train + size_eval)
    last_names_r = np.random.choice(last_names, size=size_train + size_eval)

    names = [
        f"{first_name} {last_name}"
        for first_name, last_name in zip(first_names_r, last_names_r)
    ]

    raw_ds = {"train": names[:size_train], "validation": names[size_train:]}
    tokenized_ds = {k: tokenizer(v, padding="longest") for k, v in raw_ds.items()}

    for k, v in tokenized_ds.items():
        v["labels"] = v["input_ids"].copy()

    tokenized_ds = {k: datasets.Dataset.from_dict(v) for k, v in tokenized_ds.items()}
    return tokenized_ds


def make_huggingface_ds(name, max_seq_length, text_name="text"):
    raw_dataset = load_dataset(*name)
    raw_dataset["train"] = load_dataset(*name, split="train[5%:]")
    raw_dataset["validation"] = load_dataset(*name, split="train[:5%]")

    # TODO these cells should be commented out to run on full dataset
    raw_dataset["train"] = raw_dataset["train"].select(range(200))
    raw_dataset["validation"] = raw_dataset["validation"].select(range(20))

    def tokenize_function(examples):
        out = tokenizer(
            examples[text_name],
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )
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
        #     k: [t[i : i + max_seq_length] for i in
        #     range(0, total_length, max_seq_length)]
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
    dataset = make_huggingface_ds(name=name, max_seq_length=128, text_name="review")
    return dataset


def count_params(params):
    param_count = sum(x.size for x in jax.tree_leaves(params))
    return param_count


@dataclass
class TrainingHyperparams:
    mode: str = "sgd"
    batch_size: int = 16
    num_epochs: int = 10
    training_seed: int = 0
    learning_rate: float = 3e-4

    optimizer_seed: int = 1
    noise_multiplier: float = 1.0
    l2_norm_clip: float = 1e10


### main training function:


def do_lora_training(
    filter_fn,
    lora_rank,
    save_folder,
    tokenized_datasets,
    hyperparams: TrainingHyperparams,
):
    os.makedirs(save_folder, exist_ok=True)
    print(f"--- saving to {save_folder} ---")
    try:
        t0 = time.perf_counter()

        lora_bert = LoRA(
            target_module=wrapped_model,
            pretrained_params=model.params,
            filter_fn=filter_fn,
            r=lora_rank,
        )

        ### train the LoRA model with the dataset

        num_train_steps = (
            len(tokenized_datasets["train"])
            // hyperparams.batch_size
            * hyperparams.num_epochs
        )

        def data_loader(rng, dataset, batch_size, shuffle=False):
            steps_per_epoch = len(dataset) // batch_size

            if shuffle:
                batch_idx = jax.random.permutation(rng, len(dataset))
            else:
                batch_idx = jnp.arange(len(dataset))

            batch_idx = batch_idx[
                : steps_per_epoch * batch_size
            ]  # Skip incomplete batch.
            batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

            for idx in batch_idx:
                batch = dataset[idx]
                batch = {k: jnp.array(v) for k, v in batch.items()}

                yield batch

        init_rng = jax.random.PRNGKey(0)

        init_loader = data_loader(
            init_rng, tokenized_datasets["train"], hyperparams.batch_size, shuffle=True
        )

        init_batch = next(init_loader)
        init_batch.pop("labels")
        lora_param = lora_bert.init(jax.random.PRNGKey(0), init_batch)

        print(
            f"""training 
- {count_params(lora_param)} compared to 
- {count_params(model.params)} original params"""
        )

        def loss_fn(params, batch, dropout_rng):
            labels = batch.pop("labels")
            logits = lora_bert.apply(
                variables=params, input=batch, dropout_rng=dropout_rng, train=True
            )[0]

            loss = optax.softmax_cross_entropy(
                logits[..., :-1, :], jax.nn.one_hot(labels[..., 1:], logits.shape[-1])
            ).mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)

        if hyperparams.mode == "dpsgd":

            def grad_fn_dpsgd(params, batch, dropout_rng):
                vmapped_grad_fn = jax.vmap(grad_fn, in_axes=(None, 0, None))
                batch = jax.tree_util.tree_map(lambda arr: arr[None, :], batch)
                losses, grads = vmapped_grad_fn(params, batch, dropout_rng)
                return losses.mean(), grads

            tx = optax.chain(
                optax.differentially_private_aggregate(
                    l2_norm_clip=hyperparams.l2_norm_clip,
                    noise_multiplier=hyperparams.noise_multiplier,
                    seed=hyperparams.optimizer_seed,
                ),
                optax.sgd(learning_rate=hyperparams.learning_rate),
            )
            state = train_state.TrainState.create(
                apply_fn=grad_fn_dpsgd, params=lora_param, tx=tx
            )

        else:
            if hyperparams.mode == "sgd":
                tx = optax.sgd(
                    learning_rate=hyperparams.learning_rate,
                )

            elif hyperparams.mode == "adamw":
                linear_decay_lr_schedule_fn = optax.linear_schedule(
                    init_value=hyperparams.learning_rate,
                    end_value=0,
                    transition_steps=num_train_steps,
                )
                tx = optax.adamw(
                    learning_rate=linear_decay_lr_schedule_fn,
                    b1=0.9,
                    b2=0.98,
                    eps=1e-8,
                    weight_decay=0.01,
                )
            else:
                raise ValueError

            state = train_state.TrainState.create(
                apply_fn=grad_fn, params=lora_param, tx=tx
            )

        @jax.jit
        def train_step(state, batch, dropout_rng):
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

            loss, grad = state.apply_fn(state.params, batch, dropout_rng)
            new_state = state.apply_gradients(grads=grad)

            metrics = {"loss": loss}

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

        rng = jax.random.PRNGKey(hyperparams.training_seed)
        dropout_rngs = rng

        eval_file = os.path.join(save_folder, "eval_metric.csv")
        train_file = os.path.join(save_folder, "train_metric.csv")

        for epoch in tqdm(
            range(1, hyperparams.num_epochs + 1),
            desc="Epoch ...",
            position=0,
            leave=True,
        ):
            rng, input_rng = jax.random.split(rng)

            # -- Train --
            train_loader = data_loader(
                input_rng,
                tokenized_datasets["train"],
                hyperparams.batch_size,
                shuffle=True,
            )

            with tqdm(
                total=len(tokenized_datasets["train"]) // hyperparams.batch_size,
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
                    f"Train... ({epoch}/{hyperparams.num_epochs} | Loss:"
                    f" {round(train_metric['loss'].mean(), 3)})"
                )
                t1 = time.perf_counter() - t0
                train_metrics_save = {
                    "time": t1,
                    "loss": train_metric["loss"],
                    "epoch": epoch,
                }
                train_metrics_save = pd.DataFrame.from_records([train_metrics_save])
                train_metrics_save.to_csv(
                    train_file,
                    index=False,
                    mode="a",
                    header=not os.path.exists(train_file),
                )
            # -- Eval --
            eval_loader = data_loader(
                input_rng, tokenized_datasets["validation"], hyperparams.batch_size
            )
            eval_metrics = []

            with tqdm(
                total=len(tokenized_datasets["validation"]) // hyperparams.batch_size,
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
                    f"Eval... ({epoch}/{hyperparams.num_epochs} | Loss:"
                    f" {eval_metrics['loss']} | Perplexity:"
                    f" {eval_metrics['perplexity']})"
                )

                t2 = time.perf_counter() - t0
                eval_metrics_save = {
                    "time": t2,
                    "loss": eval_metrics["loss"],
                    "perplexity": eval_metrics["perplexity"],
                }
                eval_metrics_save = pd.DataFrame.from_records([eval_metrics_save])
                eval_metrics_save.to_csv(
                    eval_file,
                    index=False,
                    mode="a",
                    header=not os.path.exists(eval_file),
                )

    except Exception as err:
        raise
        logger.error(err)


if __name__ == "__main__":
    tokenized_datasets = make_names(size_train=1000, size_eval=100)

    def filter_fn1(param_name, params):
        return "kernel" in param_name

    # hyperparams = TrainingHyperparams(
    #     mode="sgd",
    #     batch_size=16,
    #     num_epochs=10,
    #     training_seed=0,
    #     learning_rate=3e-4,
    #     optimizer_seed=1,
    # )
    #
    # do_lora_training(
    #     filter_fn=filter_fn1,
    #     lora_rank=4,
    #     save_folder=f"./exp_name/exp_sgd/",
    #     tokenized_datasets=tokenized_datasets,
    #     hyperparams=hyperparams,
    # )

    for l2_norm_clip in [1.0, 10.0, 100.0]:
        for noise_multiplier in [0.0, 0.5, 1.0, 2.0]:
            hyperparams = TrainingHyperparams(
                mode="dpsgd",
                batch_size=16,
                num_epochs=10,
                training_seed=0,
                learning_rate=3e-4,
                optimizer_seed=1,
                noise_multiplier=noise_multiplier,
                l2_norm_clip=l2_norm_clip,
            )

            do_lora_training(
                filter_fn=filter_fn1,
                lora_rank=4,
                save_folder=f"./exp_name/exp_dp_{noise_multiplier}_{l2_norm_clip}/",
                tokenized_datasets=tokenized_datasets,
                hyperparams=hyperparams,
            )

    ### exp 2
    tokenized_datasets = make_reviews()

    hyperparams = TrainingHyperparams(
        mode="sgd",
        batch_size=16,
        num_epochs=10,
        training_seed=0,
        learning_rate=3e-4,
        optimizer_seed=1,
    )

    do_lora_training(
        filter_fn=filter_fn1,
        lora_rank=4,
        save_folder="./exp_review/exp_sgd/",
        tokenized_datasets=tokenized_datasets,
        hyperparams=hyperparams,
    )

    for l2_norm_clip in [1.0, 10.0, 100.0]:
        for noise_multiplier in [0.0, 0.5, 1.0, 2.0]:
            hyperparams = TrainingHyperparams(
                mode="dpsgd",
                batch_size=16,
                num_epochs=10,
                training_seed=0,
                learning_rate=3e-4,
                optimizer_seed=1,
                noise_multiplier=noise_multiplier,
                l2_norm_clip=l2_norm_clip,
            )

            do_lora_training(
                filter_fn=filter_fn1,
                lora_rank=4,
                save_folder=f"./exp_review/exp_dp_{noise_multiplier}_{l2_norm_clip}/",
                tokenized_datasets=tokenized_datasets,
                hyperparams=hyperparams,
            )

    assert False

    ### exp 3
    tokenized_datasets = make_oscar()

    r = 2

    do_lora_training(
        filter_fn=filter_fn1,
        lora_rank=r,
        save_folder="./exp3",
        hyperparams=hyperparams,
    )
