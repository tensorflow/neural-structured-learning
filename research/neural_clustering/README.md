# Neural Clustering Library

This project contains a Tensorflow 2 implementation of Neural Clustering Process
(NCP) [1], a neural network model for supervised amortized clustering.

## How to run

```
pip install -r requirements.txt

pip install 'git+https://github.com/tensorflow/neural-structured-learning.git#egg=neural_clustering&subdirectory=research/neural_clustering'
```

This [notebook](examples/ncp_demo_with_mog_data.ipynb) demonstrates how to train
a neural clustering model and use it to cluster new datasets.

## References

[[1] A. Pakman, Y. Wang, C. Mitelut, J. Lee, L. Paninski. "Neural Clustering
Processes." ICML 2020](https://arxiv.org/abs/1901.00409)

A PyTorch implementation for NCP is also available at
https://github.com/aripakman/neural_clustering_process
