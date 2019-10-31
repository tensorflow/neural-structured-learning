# GAM: Graph Agreement Models for Semi-Supervised Learning

Neural structured learning methods such as Neural Graph Machines [1], Graph
Convolutional Networks [2] and their variants have successfully combined the
expressiveness of neural networks with graph structures to improve on learning
tasks. Graph Agreement Models (GAM) is a technique that can be applied to these
methods to handle the noisy nature of real-world graphs. Traditional graph-based
algorithms, such as label propagation, were designed with the underlying
assumption that the label of a node can be imputed from that of the neighboring
nodes and edge weights. However, most real-world graphs are either noisy or have
edges that do not correspond to label agreement uniformly across the graph.
Graph Agreement Models introduce an auxiliary model that predicts the
probability of two nodes sharing the same label as a learned function of their
features. This agreement model is then used when training a node classification
model by encouraging agreement only for those pairs of nodes that it deems
likely to have the same label, thus guiding its parameters to a better local
optima. The classification and agreement models are trained jointly in a
co-training fashion.

This code repository contains an implementation of Graph Agreement Models [3].
The code is organized into the following folders:

*   data: Classes and methods for accessing semi-supervised learning datasets.
*   models: Classes and methods for classification models and graph agreement
    models.
*   trainer: Classes and methods for training the classification models, and
    agreement models individually as well as in a co-traioning fashion.
*   experiments: Python run script for training Graph Agreement Models on
    CIFAR10 and other datasets.

The implementations of Graph Agreement Models (GAMs) are provided in the `gam`
folder on a strict "as is" basis, without warranties or conditions of any kind.
Also, these implementations may not be compatible with certain TensorFlow
versions (such as 2.0 or above) or Python versions.

<<<<<<< HEAD:research/gam/README.md
## How to run
To run GAM on a graph-based dataset (e.g., Cora, Citeseer), from this folder
run:
```bash
python3.7 -m gam.experiments.run_train_gam_graph --data_path=<path_to_data>
```

To run GAM on datasets without a graph (e.g., Cifar10), from this folder run:
```bash
python3.7 -m gam.experiments.run_train_gam
```

For running on different datasets and configuration, please check the command
line flags in each of the run scripts.

## Reference
=======
## References
>>>>>>> upstream/master:research/gam/README.md

[[1] T. Bui, S. Ravi and V. Ramavajjala. "Neural Graph Learning: Training Neural
Networks Using Graphs." WSDM 2018](https://ai.google/research/pubs/pub46568.pdf)

[[2] T. Kipf and M. Welling. "Semi-supervised classification with graph
convolutional networks." ICLR 2017](https://arxiv.org/pdf/1609.02907.pdf)

[[3] O. Stretcu, K. Viswanathan, D. Movshovitz-Attias, E.A. Platanios,
A. Tomkins, S. Ravi. "Graph Agreement Models for Semi-Supervised 
Learning." NeurIPS 2019](
https://nips.cc/Conferences/2019/Schedule?showEvent=13925)