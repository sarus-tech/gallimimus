<<<<<<< HEAD
# gallimimus
=======
A minimal implementation of the synthetic-data model.

The main differences are:

- The loss and implementation for each Codec is merged in a single module
- The CodecBuilders have been removed: the sub-codecs are simply passed as instances of the codecs
- The BatchCodec has been removed. The MetaLearner model is instantiated to handle a single observation at a time. Batching for the training is handled using `vmap`.

Minor Differences:

- ListCodec has a single function to sample (instead of `autoregressive_further`)

Missing Features:

- Marginals Pretraining
>>>>>>> 1300a94 (minimal_synthetic_data)
