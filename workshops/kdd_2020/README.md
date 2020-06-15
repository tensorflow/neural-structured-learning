# Neural Structured Learning: Training neural networks with structured signals

Hands-on tutorial at [KDD 2020](https://www.kdd.org/kdd2020/).

## Tutors

*   Allan Heydon (Google Research)
*   Arjun Gopalan (Google Research)
*   Cesar Ilharco Magalhaes (Google Research)
*   Chun-Sung Ferng (Google Research)
*   Da-Cheng Juan (Google Research)
*   Georg Osang (IST Austria and Google Research)
*   George Yu (Google Research)
*   Philip Pham (Google Research)

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

Below is an outline of how our tutorial will be structured. This is subject to
minor changes.

### Introduction to NSL

We will begin the tutorial with a presentation that gives an overview of the
Neural Structured Learning framework as well as explains the benefits of
learning with structure.

### Data preprocessing in NSL

This part of the tutorial will include a presentation discussing:

-   Graph building as a general topic including locality sensitive hashing
-   Augmenting training data for graph-based regularization in NSL
-   Related tools in the NSL framework

### Graph regularization using natural graphs

Graph regularization [2] forces neural networks to learn similar
predictions/representations for entities connected to each other in a similarity
graph. Natural graphs or organic graphs are sets of data points that have an
inherent relationship between each other. We will demonstrate via a practical
tutorial, the use of natural graphs for graph regularization for classifying the
veracity of public message posts.

### Graph regularization using synthesized graphs

Input data may not always be represented as a graph. However, one can infer
similarity relationships between entities and subsequently build a similarity
graph. We will demonstrate via a practical tutorial, the use of graph building
and subsequent graph regularization for text classification.

### Adversarial regularization

-   Practical tutorial demonstrating adversarial learning techniques [3,4] for
    image classification. It will cover methods like Fast Gradient Sign Method
    (FGSM) and Projected Gradient Descent (PGD).

### Neural Structured Learning in TensorFlow Extended (TFX)

-   Short presentation on how Neural Structured Learning can be integrated with
    [TFX](https://www.tensorflow.org/tfx) pipelines.

### Research and Future Directions

-   Presentation discussing recent research related to NSL and future directions
-   Prototype showing how NSL can be used with the
    [Graph Nets](https://github.com/deepmind/graph_nets) library.

### Closing

We will conclude our tutorial session with a presentation that will include:

-   Summary
-   Resources
-   Q/A
-   Survey/feedback

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
