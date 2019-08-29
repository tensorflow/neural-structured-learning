# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Input layer for to unpack neighbor features for graph regularization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
import neural_structured_learning.configs as configs
from neural_structured_learning.lib import utils
import tensorflow as tf


def make_missing_neighbor_inputs(neighbor_config,
                                 inputs,
                                 weight_dtype=tf.float32):
  """Makes additional inputs for neighbor features if necessary.

  Args:
    neighbor_config: An instance of `configs.GraphNeighborConfig` specifying the
      number of neighbors and how neighbor features should be named.
    inputs: Dictionary of input tensors that may be missing neighbor features.
      The keys are the features names. See `utils.unpack_neighbor_features` for
      expected names of neighbor features and weights.
    weight_dtype: `tf.Dtype` for neighbors weights. Defaults to `tf.float32`.

  Returns:
    A dictionary of neighbor feature and weight tensors that do not already
    exist in `inputs`. The keys are specified according to `neighbor_config`.
  """
  existing_feature_names = set(inputs.keys())
  neighbor_inputs = {}
  for i in range(neighbor_config.max_neighbors):  # For each potential neighbor.
    # Weight of the neighbor.
    weight_name = '{}{}{}'.format(neighbor_config.prefix, i,
                                  neighbor_config.weight_suffix)
    if weight_name not in existing_feature_names:
      neighbor_inputs[weight_name] = tf.keras.Input((1,),
                                                    dtype=weight_dtype,
                                                    name=weight_name)
    # For inputs without existing neighbor features, replicate them.
    for feature_name, tensor in inputs.items():
      if feature_name.startswith(neighbor_config.prefix):
        continue
      neighbor_feature_name = '{}{}_{}'.format(neighbor_config.prefix, i,
                                               feature_name)
      if neighbor_feature_name not in existing_feature_names:
        neighbor_inputs[neighbor_feature_name] = tf.keras.Input(
            tensor.shape[1:],
            batch_size=tensor.shape[0],
            dtype=tensor.dtype,
            name=neighbor_feature_name,
            ragged=isinstance(tensor, tf.RaggedTensor),
            sparse=isinstance(tensor, tf.sparse.SparseTensor))
  return neighbor_inputs


class NeighborFeatures(tf.keras.layers.Layer):
  """A layer to unpack a dictionary of sample features and neighbor features.

  Missing neighbor inputs will also be created for the functional API.
  """

  def __init__(self,
               neighbor_config=None,
               feature_names=None,
               weight_dtype=None,
               **kwargs):
    """Initializes an instance of `NeighborFeatures`.

    Args:
      neighbor_config: A `configs.GraphNeighborConfig` instance describing
        neighbor attributes.
      feature_names: Optional[List[Text]], names denoting the keys of features
        for which to create neighbor inputs. If `None`, all features are assumed
        to have corresponding neighbor features.
      weight_dtype: `tf.DType` for `neighbor_weights`. Defaults to `tf.float32`.
      **kwargs: Additional arguments to be passed `tf.keras.layers.Layer`.
    """
    super(NeighborFeatures, self).__init__(
        autocast=False,
        dtype=kwargs.pop('dtype') if 'dtype' in kwargs else weight_dtype,
        **kwargs)
    self._neighbor_config = (
        configs.GraphNeighborConfig()
        if neighbor_config is None else attr.evolve(neighbor_config))
    self._feature_names = (
        feature_names if feature_names is None else set(feature_names))

  def call(self, inputs, keep_rank=False):
    """Extracts neighbor features and weights from a dictionary of inputs.

    This function is a wrapper around `utils.unpack_neighbor_features`. See
    `utils.unpack_neighbor_features` for documentation on the expected input
    format and return values.

    Args:
      inputs: Dictionary of `tf.Tensor` features with keys for neighbors and
        weights described by `neighbor_config`.
      keep_rank: Defaults to `False`. If `True`, each value of
        `neighbor_features` will have an extra neighborhood size dimension at
        axis 1.

    Returns:
      A tuple (sample_features, neighbor_features, neighbor_weights) of tensors.
      See `utils.unpack_neighbor_features` for a detailed description.
    """
    return utils.unpack_neighbor_features(
        inputs, self._neighbor_config, keep_rank=keep_rank)

  def _include_feature(self, name):
    """Decides if the feature specified by `name` should be a model input."""
    return (self._feature_names is None or name in self._feature_names or
            name.startswith(self._neighbor_config.prefix))

  def __call__(self, inputs, *args, **kwargs):
    """Calls the layer and updates `inputs` with new features if necessary.

    Args:
      inputs: A dictionary of tensors keyed by their feature names. See
        `utils.unpack_neighbor_features` for expected names of neighbor feature
        and weight tensors. If `inputs` is missing any neighbor feature and
        weight tensors, the dictionary will be updated with additional inputs
        corresponding to neighbor features and weights. These additional inputs
        should be passed to `tf.keras.Model` when using the functional API.
      *args: Positional arguments forwarded to `call` and `Layer.__call__`.
      **kwargs: Keyword arguments forwarded to `call` and `Layer.__call__`.

    Returns:
      A tuple (sample_features, neighbor_features, neighbor_weights) of tensors.
      See `utils.unpack_neighbor_features` for a detailed description.
    """
    filtered_inputs = {
        feature_name: feature_tensor
        for feature_name, feature_tensor in inputs.items()
        if self._include_feature(feature_name)
    }
    missing_neighbor_inputs = make_missing_neighbor_inputs(
        self._neighbor_config, filtered_inputs, weight_dtype=self.dtype)
    # Mutate `inputs` for Functional API.
    inputs.update(missing_neighbor_inputs)
    filtered_inputs.update(missing_neighbor_inputs)
    # Only unpack the relevant inputs.
    return super(NeighborFeatures, self).__call__(filtered_inputs, *args,
                                                  **kwargs)

  def get_config(self):
    config = super(NeighborFeatures, self).get_config()
    config['neighbor_config'] = attr.asdict(self._neighbor_config)
    config['feature_names'] = (
        list(self._feature_names) if self._feature_names is not None else None)
    return config

  @classmethod
  def from_config(cls, config):
    return cls(
        configs.GraphNeighborConfig(**config.pop('neighbor_config')), **config)
