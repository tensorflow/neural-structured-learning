# Neural Structured Learning: Training Neural Networks with Structured Signals

Hands-on tutorial at
[WSDM 2021](https://www.wsdm-conference.org/2021/tutorials.php/).

## Organizers

*   Allan Heydon (Google Research)
*   Arjun Gopalan (Google Research)
*   Cesar Ilharco Magalhaes (Google Research)
*   Chun-Sung Ferng (Google Research)
*   Chun-Ta Lu (Google Research)
*   Da-Cheng Juan (Google Research)
*   George Yu (Google Research)
*   Philip Pham (Google Research)
*   Yicheng Fan (Google Research)
*   Yueqi Wang (Google Research)

## Overview

This tutorial will cover several aspects of
[Neural Structured Learning](https://www.tensorflow.org/neural_structured_learning)
(NSL) [1] with an emphasis on two techniques -- graph regularization and
adversarial regularization. In addition to using interactive hands-on tutorials
that demonstrate the NSL framework and APIs in TensorFlow, we also plan to have
short presentations to provide additional context and motivation. Further, we
will discuss some recent research that is closely related to Neural Structured
Learning but not yet part of the framework in TensorFlow. Topics here will
include using graphs for learning embeddings [5,6] and other advanced models of
graph neural networks [7,8].

## Outline

Below is the outline of our tutorial.

### Introduction to NSL

We will begin the tutorial with an overview of the Neural Structured Learning
framework and motivate the advantages of training neural networks with
structured signals.

[[Slides](slides/Introduction.pdf)]

### Data preprocessing in NSL

This part of the tutorial will include a presentation discussing:

-   Graph building as a general topic including locality sensitive hashing
-   Augmenting training data for graph-based regularization in NSL
-   Related tools in the NSL framework

[[Slides](slides/Data_Preprocessing.pdf)]

### Graph regularization using natural graphs (Lab 1)

Graph regularization [2] forces neural networks to learn similar
predictions/representations for entities connected to each other in a similarity
graph. *Natural graphs* or *organic graphs* are sets of data points that have an
inherent relationship between each other. We will demonstrate via a practical
tutorial, the use of natural graphs for graph regularization to classify the
veracity of public message posts.

[[Slides](slides/Natural_Graphs.pdf)]
[[Colab tutorial](https://colab.research.google.com/github/tensorflow/neural-structured-learning/blob/master/workshops/kdd_2020/graph_regularization_pheme_natural_graph.ipynb)]

### Graph regularization using synthesized graphs (Lab 2)

Input data may not always be represented as a graph. However, one can infer
similarity relationships between entities and subsequently build a similarity
graph. We will demonstrate graph building and subsequent graph regularization
for text classification using a practical tutorial. While graphs can be built in
many ways, we will make use of text embeddings in this tutorial to build a
graph.

[[Slides](slides/Synthesized_Graphs.pdf)]
[[Colab tutorial](https://colab.research.google.com/github/tensorflow/neural-structured-learning/blob/master/g3doc/tutorials/graph_keras_lstm_imdb.ipynb)]

### Adversarial regularization (Lab 3)

Adversarial learning has been shown to be effective in improving the accuracy of
a model as well as its robustness to adversarial attacks. We will demonstrate
adversarial learning techniques [3,4] like *Fast Gradient Sign Method* (FGSM)
and *Projected Gradient Descent* (PGD) for image classification using a
practical tutorial.

[[Slides](slides/Adversarial_Learning.pdf)]
[[Colab tutorial](https://colab.research.google.com/github/tensorflow/neural-structured-learning/blob/master/workshops/kdd_2020/adversarial_regularization_mnist.ipynb)]

### Scalable graph learning with historical embeddings

We will describe a new framework for scalable graph learning and demonstrate a
prototype implementation. The framework uses the concept of historical
embeddings i.e, embeddings computed and stored in a previous training step, to
make the training time nearly invariant to the size of local neighborhoods. This
also enables the use of dynamic graphs and neighborhoods for graph learning.

[[Slides](slides/CARLS.pdf)]

### Neural Clustering Process (Lab 4)

In this session, we will demonstrate supervised neural clustering methods [9,
10], such as Neural Clustering Process (NCP), that can be trained end-to-end to
perform probabilistic clustering on both generic data points and graph
structured data without assuming a fixed or maximum number of clusters.

[[Slides](slides/Neural_Clustering.pdf)]
[[Colab tutorial](https://github.com/tensorflow/neural-structured-learning/blob/master/research/neural_clustering/examples/ncp_demo_with_mog_data.ipynb)]

### Research and Future Directions

In this session, we will discuss:

    -   Recent research related to NSL
    -   Building graph neural networks with NSL and the
        [Graph Nets](https://github.com/deepmind/graph_nets) [11] library
    -   Future directions for NSL research and development
    -   Academic and industrial collaboration opportunities

Finally, we will conclude our tutorial with a summary of the entire session,
provide links to various NSL resources, and share a link to a brief survey to
get feedback on the NSL framework and the hands-on tutorial.

[[Slides](slides/Research_and_Future_Directions.pdf)]

## References

1.  https://www.tensorflow.org/neural_structured_learning
2.  T. Bui, S. Ravi, V. Ramavajjala, “Neural Graph Learning: Training Neural
    Networks Using Graphs,” WSDM 2018.
3.  I. Goodfellow, J. Shlens, C. Szegedy, “Explaining and Harnessing Adversarial
    Examples,” ICLR 2015
4.  T. Miyato, S. Maeda, M. Koyama and S. Ishii, “Virtual Adversarial Training:
    A Regularization Method for Supervised and Semi-Supervised Learning,” IEEE
    Transactions on Pattern Analysis and Machine Intelligence 2019.
5.  D.C. Juan, C.T. Lu, Z. Li, F. Peng, A. Timofeev, Y.T. Chen, Y. Gao, T.
    Duerig, A. Tomkins and S. Ravi, “Ultra Fine-Grained Image Semantic
    Embedding,” WSDM 2020
6.  T. Bansal, D.C. Juan, S. Ravi, A. McCallum, “A2N: Attending to Neighbors for
    Knowledge Graph Inference,” ACL 2019
7.  O. Stretcu, K. Viswanathan, D. Movshovitz-Attias, E.A. Platanios, S. Ravi,
    A. Tomkins. "Graph Agreement Models for Semi-Supervised Learning," NeurIPS
    2019
8.  Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, P. Yu, “A Comprehensive Survey on
    Graph Neural Networks” arXiv 2019.
9.  A. Pakman, Y. Wang, Y. Lee, P. Basu, J. Lee, Y.W. Teh, L. Paninski.
    “Attentive Clustering Processes“ arXiv 2020.
10. A. Pakman, Y. Wang, C. Mitelut, J. Lee, L. Paninski. ”Neural Clustering
    Processes” PMLR 2020
11. https://github.com/deepmind/graph_nets
