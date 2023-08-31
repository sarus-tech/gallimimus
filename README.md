An implementation of [LoRA](https://arxiv.org/abs/2106.09685) in Flax.

### Installation

`pip install .` or `pip install -r requirements-dev.txt` to install lora-flax.

As demonstrated in the examples in `testing/`, lora-flax takes a Flax module and 
pretrained weights and builds a new Flax module with the same interface , but where the
pretrained parameters are frozen, and the interactive parameters (those given by `.init`
and used in `.apply`) are now the LoRA parameters.