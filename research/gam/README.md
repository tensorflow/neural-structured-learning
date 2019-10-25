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

This code repository contains an implementation of Graph Agreement Models. The
code is organized into the following folders:

*   data: Classes and methods for accessing semi-supervised learning datasets.
*   models: Classes and methods for classification models and graph agreement
    models.
*   trainer: Classes and methods for training the classification models, and
    agreement models individually as well as in a co-traioning fashion.
*   experiments: Python run script for training Graph Agreement Models on MNIST
    and other datasets.

The implementations of Graph Agreement Models (GAMs) are provided in the `gam`
folder on a strict "as is" basis, without warranties or conditions of any kind.
Also, these implementations may not be compatible with certain TensorFlow
versions (such as 2.0 or above) or Python versions.

## References

[[1] T. Bui, S. Ravi and V. Ramavajjala. "Neural Graph Learning: Training Neural
Networks Using Graphs." WSDM 2018](https://ai.google/research/pubs/pub46568.pdf)

[[2] T. Kipf and M. Welling. "Semi-supervised classification with graph
convolutional networks." ICLR 2017](https://arxiv.org/pdf/1609.02907.pdf)
