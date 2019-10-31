# A2N: Attending to Neighbors for Knowledge Graph Inference

State-of-the-art models for knowledge graph completion aim at learning a fixed
embedding representation of entities in a multi-relational graph which can
generalize to infer unseen entity relationships at test time. This can be
sub-optimal as it requires memorizing and generalizing to all possible entity
relationships using these fixed representations. We thus propose a novel
attention-based method to learn query-dependent representation of entities which
adaptively combines the relevant graph neighborhood of an entity leading to more
accurate KG completion. The proposed method is evaluated on two benchmark
datasets for knowledge graph completion, and experimental results show that the
proposed model performs competitively or better than existing state-of-the-art,
including recent methods for explicit multi-hop reasoning. Qualitative probing
offers insight into how the model can reason about facts involving multiple hops
in the knowledge graph, through the use of neighborhood attention.

The implementations of A2N [1] are provided on a strict "as is" basis, without
warranties or conditions of any kind. Also, these implementations may not be
compatible with certain TensorFlow versions (such as 2.0 or above) or Python
versions.

## References

[[1] T. Bansal, D. Juan, S. Ravi and A. McCallum. "A2N: Attending to Neighbors
for Knowledge Graph Inference." ACL
2019](https://www.aclweb.org/anthology/P19-1431)
