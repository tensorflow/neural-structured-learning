# GAM: Graph Agreement Models for Semi-Supervised Learning

This code repository contains an implementation of Graph Agreement Models [1].

Neural structured learning methods such as Neural Graph Machines [2], Graph
Convolutional Networks [3] and their variants have successfully combined the
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

The code is organized into the following folders:

*   data: Classes and methods for accessing semi-supervised learning datasets.
*   models: Classes and methods for classification models and graph agreement
    models.
*   trainer: Classes and methods for training the classification models, and
    agreement models individually as well as in a co-training fashion.
*   experiments: Python run script for training Graph Agreement Models on
    CIFAR10 and other datasets.

The implementations of Graph Agreement Models (GAMs) are provided in the `gam`
folder on a strict "as is" basis, without warranties or conditions of any kind.
Also, these implementations may not be compatible with certain TensorFlow
versions (such as 2.0 or above) or Python versions.

More details can be found in our
[paper](https://papers.nips.cc/paper/9076-graph-agreement-models-for-semi-supervised-learning.pdf),
[supplementary material](https://papers.nips.cc/paper/9076-graph-agreement-models-for-semi-supervised-learning-supplemental.zip),
[slides](https://drive.google.com/open?id=1tWEMoyrbLnzfSfTfYFi9eWgZWaPKF3Uu) or
[poster](https://drive.google.com/file/d/1BZNR4B-xM41hdLLqx4mLsQ4KKJOhjgqV/view).

## Updated Results

A bug was discovered in the implementation of the GAM agreement regularization
term after publication. We have fixed the bug (PR #82) and have rerun the
affected experiments. Below are the updated results (note that the GAM* results
are not affected).

<img src="img/gam-updated-results.png " width="400">

Although some of these numbers are lower than what was originally reported, the
takeaways presented in our paper still hold: GAM adds a significant boost to the
original base models, and also performs better than other forms of
regularization reported in our paper. Nevertheless, we apologize for any
inconvenience caused by this bug!

## How to run

To run GAM on a graph-based dataset (e.g., Cora, Citeseer, Pubmed), from this
folder run: `$ python3.7 -m gam.experiments.run_train_gam_graph
--data_path=<path_to_data>`

To run GAM on datasets without a graph (e.g., CIFAR10), from this folder run: `$
python3.7 -m gam.experiments.run_train_gam`

We recommend running on a GPU. With CUDA, this can be done by prepending
`CUDA_VISIBLE_DEVICES=<your-gpu-number>` in front of the run command.

For running on different datasets and configuration, please check the command
line flags in each of the run scripts. The configurations used in our paper can
be found in the file `run_configs.txt`.

## Visualizing the results.

To visualize the results in Tensorboard, use the following command, adjusting
the dataset name accordingly: `$ tensorboard --logdir=outputs/summaries/cora`

An example of such visualization for Cora with GCN + GAM model on the Pubmed
dataset is the following:
![Tensorboard plot](img/gam_gcn_pubmed.png?raw=true "GCN + GAM on Pubmed")

Similarly, we can run with multiple different parameter configurations and plot
the results together for comparison. An example showing the accuracy per
co-train iteration of a GCN + GAM model on the Cora dataset for 3 runs with 3
different random seeds is the following:
![Tensorboard plot](img/gam_gcn_cora_multiple_seeds.png?raw=true "GCN + GAM on Cora")

## References

[[1] O. Stretcu, K. Viswanathan, D. Movshovitz-Attias, E.A. Platanios, S. Ravi,
A. Tomkins. "Graph Agreement Models for Semi-Supervised Learning." NeurIPS
2019](https://papers.nips.cc/paper/9076-graph-agreement-models-for-semi-supervised-learning)

[[2] T. Bui, S. Ravi and V. Ramavajjala. "Neural Graph Learning: Training Neural
Networks Using Graphs." WSDM 2018](https://research.google/pubs/pub46568.pdf)

[[3] T. Kipf and M. Welling. "Semi-supervised classification with graph
convolutional networks." ICLR 2017](https://arxiv.org/pdf/1609.02907.pdf)
