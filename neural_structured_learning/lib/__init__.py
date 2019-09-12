"""Library APIs for Neural Structured Learning."""

from neural_structured_learning.lib.abstract_gen_neighbor import GenNeighbor
from neural_structured_learning.lib.adversarial_neighbor import gen_adv_neighbor
from neural_structured_learning.lib.distances import jensen_shannon_divergence
from neural_structured_learning.lib.distances import kl_divergence
from neural_structured_learning.lib.distances import pairwise_distance_wrapper
from neural_structured_learning.lib.regularizer import adv_regularizer
from neural_structured_learning.lib.regularizer import virtual_adv_regularizer
from neural_structured_learning.lib.utils import apply_feature_mask
from neural_structured_learning.lib.utils import decay_over_time
from neural_structured_learning.lib.utils import get_target_indices
from neural_structured_learning.lib.utils import maximize_within_unit_norm
from neural_structured_learning.lib.utils import normalize
from neural_structured_learning.lib.utils import replicate_embeddings
from neural_structured_learning.lib.utils import unpack_neighbor_features

__all__ = [
    'adv_regularizer',
    'apply_feature_mask',
    'decay_over_time',
    'GenNeighbor',
    'gen_adv_neighbor',
    'get_target_indices',
    'jensen_shannon_divergence',
    'kl_divergence',
    'maximize_within_unit_norm',
    'normalize',
    'pairwise_distance_wrapper',
    'replicate_embeddings',
    'unpack_neighbor_features',
    'virtual_adv_regularizer',
]
