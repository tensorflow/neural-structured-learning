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
"""Example of Graph Regularization with an NSL GNN."""
import functools

from absl import app
from absl import flags
import graph_nets
import neural_structured_learning as nsl
from neural_structured_learning.experimental import gnn
import tensorflow as tf

flags.DEFINE_string(
    'train_examples_path',
    None,
    'Path to training examples.')
flags.DEFINE_string('eval_examples_path',
                    None,
                    'Path to evaluation examples.')

FLAGS = flags.FLAGS


class NodeClassifier(tf.keras.Model):
  """Classifier model for nodes."""

  def __init__(self,
               seq_length,
               num_classes,
               hidden_units=None,
               dropout_rate=0.5,
               **kwargs):
    inputs = tf.keras.Input(shape=(seq_length,), dtype=tf.int64, name='words')
    x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(inputs)
    for num_units in (hidden_units or [50, 50]):
      x = tf.keras.layers.Dense(num_units, activation='relu')(x)
      x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    super(NodeClassifier, self).__init__(inputs, outputs, **kwargs)


def main(argv):
  del argv
  graph_reg_config = nsl.configs.GraphRegConfig(
      neighbor_config=nsl.configs.GraphNeighborConfig(max_neighbors=3),
      multiplier=0.1,
      distance_config=nsl.configs.DistanceConfig(
          distance_type=nsl.configs.DistanceType.L2,
          reduction=tf.compat.v1.losses.Reduction.NONE,
          sum_over_axis=-1))

  train_dataset = gnn.make_cora_dataset(
      FLAGS.train_examples_path,
      shuffle=True,
      neighbor_config=graph_reg_config.neighbor_config)
  eval_dataset = gnn.make_cora_dataset(FLAGS.eval_examples_path, batch_size=32)

  model = gnn.GraphRegularizationModel(
      config=graph_reg_config,
      node_model_fn=functools.partial(
          NodeClassifier,
          seq_length=tf.data.experimental.get_structure(train_dataset)[0]
          ['words'].shape[-1],
          num_classes=7))
  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[
          tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
          tf.keras.metrics.SparseCategoricalAccuracy(),
          tf.keras.metrics.SparseTopKCategoricalAccuracy(2),
      ])
  model.fit(train_dataset, epochs=30, validation_data=eval_dataset)


if __name__ == '__main__':
  graph_nets.compat.set_sonnet_version('2')
  app.run(main)
