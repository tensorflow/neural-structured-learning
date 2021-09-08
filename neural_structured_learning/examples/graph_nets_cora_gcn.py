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
"""Example of an NSL GNN."""
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


def main(argv):
  del argv
  neighbor_config = nsl.configs.GraphNeighborConfig(max_neighbors=3)
  train_dataset = gnn.make_cora_dataset(
      FLAGS.train_examples_path, shuffle=True, neighbor_config=neighbor_config)
  eval_dataset = gnn.make_cora_dataset(FLAGS.eval_examples_path, batch_size=32)

  model = gnn.GraphConvolutionalNodeClassifier(
      seq_length=tf.data.experimental.get_structure(train_dataset)[0]
      ['words'].shape[-1],
      num_classes=7)
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
