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
"""Experimental implementation integrating GraphNets into NSL."""
import functools
from typing import Dict, Optional, Text

import graph_nets
import neural_structured_learning.configs as configs
import neural_structured_learning.keras as keras
import neural_structured_learning.lib as lib
import sonnet.v2 as snt
import tensorflow as tf

graph_nets.compat.set_sonnet_version('2')


def _make_ragged(batch_size: int, num_neighbors: int, x: tf.Tensor):
  return tf.RaggedTensor.from_row_lengths(x, [num_neighbors] * batch_size)


def pack_nodes_and_edges(batch_size: int,
                         neighbor_config: configs.GraphRegConfig,
                         features: Dict[Text, tf.Tensor], *args):
  """Pack features into a graph."""
  if neighbor_config.max_neighbors == 0:
    graph = {key: value for key, value in features.items()}
    graph['n_node'] = tf.ones(batch_size, dtype=tf.int64)
    graph['n_edge'] = tf.zeros(batch_size, dtype=tf.int64)
    graph['edges'] = tf.zeros([0, 1], dtype=tf.float32)
    graph['senders'] = graph['receivers'] = tf.zeros(0, dtype=tf.int64)
    graph['globals'] = tf.zeros([batch_size, 0], tf.float32)
    return (graph,) + args if args else graph
  ragged_fn = functools.partial(_make_ragged, batch_size,
                                neighbor_config.max_neighbors)
  features, neighbor_features, neighbor_weights = (
      keras.layers.NeighborFeatures(neighbor_config)(features))
  neighbor_weights = ragged_fn(tf.squeeze(neighbor_weights))
  neighbor_weights.flat_values.set_shape(
      [batch_size * neighbor_config.max_neighbors])
  mask = neighbor_weights.with_flat_values(neighbor_weights.flat_values > 0)
  neighbor_features = tf.nest.map_structure(
      lambda x: tf.ragged.boolean_mask(ragged_fn(x), mask), neighbor_features)
  neighbor_weights = tf.ragged.boolean_mask(neighbor_weights, mask)
  graph = {}
  for key, value in features.items():
    graph[key] = tf.concat([
        neighbor_features[key],
        tf.RaggedTensor.from_row_splits(
            value, tf.range(batch_size + 1, dtype=tf.int64)),
    ], 1)
  # Just take it from the last feature and assume it's all the same.
  center_node_ids = list(graph.values())[-1].row_limits() - 1
  n_node = list(graph.values())[-1].row_lengths()
  graph['edges'] = tf.reshape(neighbor_weights.flat_values, [-1, 1])
  graph['senders'] = (
      tf.range(tf.shape(neighbor_weights.flat_values)[0], dtype=tf.int64) +
      neighbor_weights.value_rowids())
  graph['receivers'] = tf.gather(center_node_ids,
                                 neighbor_weights.value_rowids())
  graph['n_edge'] = neighbor_weights.row_lengths()
  graph['n_node'] = n_node
  graph['globals'] = tf.zeros([batch_size, 0], tf.float32)
  graph = {
      key: value.flat_values if isinstance(value, tf.RaggedTensor) else value
      for key, value in graph.items()
  }
  return (graph,) + args if args else graph


def make_cora_dataset(
    file_path,
    batch_size=128,
    shuffle=False,
    neighbor_config: Optional[configs.GraphNeighborConfig] = None,
    max_seq_length=1433,
    num_parallel_calls=tf.data.experimental.AUTOTUNE):
  """Returns a `tf.data.Dataset` instance based on data in `file_path`."""
  if neighbor_config is None:
    neighbor_config = configs.GraphNeighborConfig()
  features = {
      'words':
          tf.io.FixedLenFeature([max_seq_length],
                                tf.int64,
                                default_value=tf.constant(
                                    0, dtype=tf.int64, shape=[max_seq_length])),
      'label':
          tf.io.FixedLenFeature((), tf.int64, default_value=-1),
  }
  for i in range(neighbor_config.max_neighbors):
    nbr_feature_key = '{}{}_{}'.format(neighbor_config.prefix, i, 'words')
    nbr_weight_key = '{}{}{}'.format(neighbor_config.prefix, i,
                                     neighbor_config.weight_suffix)
    features[nbr_feature_key] = tf.io.FixedLenFeature(
        [max_seq_length],
        tf.int64,
        default_value=tf.constant(0, dtype=tf.int64, shape=[max_seq_length]))
    features[nbr_weight_key] = tf.io.FixedLenFeature([1],
                                                     tf.float32,
                                                     default_value=tf.constant(
                                                         [0.0]))
  dataset = tf.data.experimental.make_batched_features_dataset(
      [file_path],
      batch_size,
      features,
      label_key='label',
      num_epochs=1,
      shuffle=shuffle,
      drop_final_batch=True)
  dataset = dataset.map(
      functools.partial(pack_nodes_and_edges, batch_size, neighbor_config),
      num_parallel_calls=num_parallel_calls)
  return dataset.prefetch(num_parallel_calls)


class NodeGraphModel(tf.keras.Model):
  """Packs features into a `graph_nets.graph.GraphsTuple`."""

  def graph_call(self, graph, **kwargs):
    raise NotImplementedError('graph_call should be implemented.')

  def call(self, inputs, **kwargs):
    # Pack the inputs into a graph. Keras may add an extra dimension. Remove it.
    graph = graph_nets.graphs.GraphsTuple(
        n_node=tf.reshape(inputs.pop('n_node'), (-1,), name='reshape_n_node'),
        n_edge=tf.reshape(inputs.pop('n_edge'), (-1,), name='reshape_n_edge'),
        globals=inputs.pop('globals'),
        edges=inputs.pop('edges'),
        senders=tf.reshape(
            inputs.pop('senders'), (-1,), name='reshape_senders'),
        receivers=tf.reshape(
            inputs.pop('receivers'), (-1,), name='reshape_receivers'),
        nodes=inputs)
    # Transform the graph.
    graph = self.graph_call(graph, **kwargs)
    # Gather up the centered nodes from the graph.
    node_ids = tf.cumsum(graph.n_node) - 1
    return tf.nest.map_structure(lambda x: tf.gather(x, node_ids), graph.nodes)


class GraphRegularizationEdgeModel(tf.keras.layers.Layer):

  def __init__(self, config: configs.DistanceConfig, **kwargs):
    super(GraphRegularizationEdgeModel, self).__init__(**kwargs)
    self._config = config

  def call(self, inputs):
    weights = inputs[:, :1]
    sources, targets = tf.split(inputs[:, 1:], 2, axis=1)
    return tf.math.reduce_sum(
        lib.pairwise_distance_wrapper(sources, targets, weights, self._config),
        -1)


class EdgeRelationNetwork(graph_nets.modules.RelationNetwork):
  """RelationNetwork that uses edge features in the the EdgeBlock."""

  @functools.wraps(graph_nets.modules.RelationNetwork.__init__)
  def __init__(self, *args, **kwargs):
    super(EdgeRelationNetwork, self).__init__(*args, **kwargs)
    self._edge_block._use_edges = True  # pylint: disable=protected-access


class GraphRegularizationModel(NodeGraphModel):
  """Model that does graph regularization given a node model."""

  def __init__(self, config: configs.GraphRegConfig, node_model_fn, **kwargs):
    super(GraphRegularizationModel, self).__init__(**kwargs)
    self._graph_loss_multiplier = config.multiplier
    # Can't do this in build since our input is a dictionary.
    self._example_weights_fn = graph_nets.blocks.EdgesToGlobalsAggregator(
        tf.math.unsorted_segment_sum)
    self._node_model = node_model_fn()
    self._edge_model = GraphRegularizationEdgeModel(config.distance_config)

  def graph_call(self, graph, **kwargs):
    # Convert the inputs to a graph and run the model.
    example_weights = tf.reshape(self._example_weights_fn(graph), (-1,))
    snt.allow_empty_variables(self._example_weights_fn)
    graph = graph_nets.modules.GraphIndependent(
        node_model_fn=lambda: self._node_model)(
            graph)
    graph = EdgeRelationNetwork(
        edge_model_fn=lambda: self._edge_model,
        global_model_fn=lambda: (lambda x: x))(
            graph)
    graph_loss = tf.math.reduce_mean(
        tf.math.divide_no_nan(graph.globals, example_weights),
        name='graph_loss')
    self.add_metric(graph_loss, aggregation='mean', name='graph_loss')
    self.add_loss(self._graph_loss_multiplier * graph_loss)
    return graph


class GraphConvolutionalNodeClassifier(NodeGraphModel):
  """Classifies nodes with a simple Graph Convolutional Network."""

  def __init__(self,
               seq_length,
               num_classes,
               hidden_units=16,
               dropout_rate=0.25,
               **kwargs):
    super(GraphConvolutionalNodeClassifier, self).__init__(**kwargs)
    self._dense_features = tf.keras.layers.DenseFeatures([
        tf.feature_column.numeric_column('words', shape=(seq_length,)),
    ])
    # First GCN block with shared edge and node encoder.
    self._node_encoder_model1 = tf.keras.layers.Dense(hidden_units)
    self._edge_model1 = self._node_encoder_model1
    self._node_model1 = tf.keras.Sequential([
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(
            hidden_units, activation=tf.keras.activations.relu),
    ])
    # Second GCN block.
    self._edge_model2 = tf.keras.layers.Dense(
        hidden_units, activation=tf.keras.activations.relu)
    self._node_encoder_model2 = tf.keras.layers.Dense(
        hidden_units, activation=tf.keras.activations.relu)
    self._node_model2 = tf.keras.Sequential([
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, name='logits'),
    ])

  def graph_call(self, graph, **kwargs):
    graph = graph_nets.modules.GraphIndependent(
        node_model_fn=lambda: self._dense_features)(
            graph)
    graph = graph_nets.modules.CommNet(
        edge_model_fn=lambda: self._edge_model1,
        node_encoder_model_fn=lambda: self._node_encoder_model1,
        node_model_fn=lambda: self._node_model1)(
            graph)
    graph = graph_nets.modules.CommNet(
        edge_model_fn=lambda: self._edge_model2,
        node_encoder_model_fn=lambda: self._node_encoder_model2,
        node_model_fn=lambda: self._node_model2)(
            graph)
    return graph
