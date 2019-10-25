# An Implementation for the optimizer [LAMB](https://arxiv.org/abs/1904.00962) with Tensorflow 2.0

We use pure python code inherted from `tensorflow.keras.optimizers.Adam` to implement the LAMB algorithm ( Layerwise Adaptive Large Latch optimization).

This implementation uses L2 regularization for all weights excepted for layernorm and all biases.
