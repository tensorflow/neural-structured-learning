# Candidate Sampling

The candidate sampling component of CARLS is responsible for sampling data from
a knowledge bank based on different application. For example

*   Sampled Softmax/Logistic for efficient loss computation
    ([TensorFlow Doc](https://www.tensorflow.org/extras/candidate_sampling.pdf)):
    this is useful when the target class in a model is very large or highly
    dynamic.

*   Top-K/Nearest Neighbors: find the top-k closest target classses in a
    softmax/logistic top layer during model inference.

*   Attention-based on external knowledge retrieval: retrieve an attention
    vector based on given input query from a sampled large set of values.
