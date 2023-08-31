# gallimimus

A minimal implementation of the synthetic data model for complex and hierarchical data. 
Documentation at [this page](https://sarus-tech.github.io/gallimimus/).

See in `testing/` an example sharing a Codec for multiple columns, as well as an example
where a pretrained language model is used.

### Installation

Do `pip install .` or `pip install -r requirements-dev.txt` to install gallimimus.

### Branches

This is the main branch of work. It contains the standard model, whose main features
is that all the models used in the generation (codec and others, for instance LLMs) are 
shared everywhere and can therefore be used in different places for the generation.

The other branches are

- `lora_flax` contains an implementation of LoRA for Flax modules. It is a requirement
    for this module, and is automatically installed.
- `interpolated_codec` implements the `RealCategoricalCodec` to handle real numbers
    by interpolating the embeddings associated to quantiles in the distribution.