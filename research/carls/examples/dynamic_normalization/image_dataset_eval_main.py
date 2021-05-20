# Copyright 2021 Google LLC
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
"""Script to evaluate DynamicNormalization on various image datasets.

To compare with BatchNormalization, set --use_batch_normalization=true.
"""

import typing
from absl import app
from absl import flags
from research.carls import dynamic_embedding_config_pb2 as de_config_pb2
from research.carls import dynamic_memory_ops as dm_ops
from research.carls import dynamic_normalization as dn
from research.carls import kbs_server_helper_pybind as kbs_server_helper
import tensorflow as tf
import tensorflow_datasets as tfds
from google.protobuf import text_format

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'dataset', 'mnist', ['mnist', 'cifar10', 'cifar100'],
    'Dataset name for training, can be one of "mnist", "cifar10", etc')

flags.DEFINE_integer('hidden_size', 128, 'Hidden layer size.')

flags.DEFINE_integer('per_cluster_buffer_size', 32,
                     'Buffer size for each cluster of DynamicMemory.')

flags.DEFINE_integer('max_num_clusters', 100,
                     'Maximal number of clusters in a DynamicMemory.')

flags.DEFINE_integer('bootstrap_steps', 1000,
                     'Number of steps to train the model without growing.')

flags.DEFINE_integer('batch_size', 16, 'Batch size for training.')

flags.DEFINE_integer('num_epochs', 5, 'Number of epochs for training.')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training.')

flags.DEFINE_float(
    'distance_to_cluster_threshold', 0.5,
    'The threshold beyond which a new cluster should be created.')

flags.DEFINE_float('min_variance', 0.001,
                   'Minimal prior variance value for each cluster.')

flags.DEFINE_boolean(
    'use_batch_normalization', False,
    'If true, use batch normalization layer instead of dynamic normalization.')


def preprocess_image(data):
  """Returns the normalized images and labels."""
  return tf.cast(data['image'], tf.float32) / 255., data['label']


def build_dm_config_from_flags():
  """Builds a DynamicEmbeddingConfig from flags."""
  config = de_config_pb2.DynamicEmbeddingConfig()
  text_format.Parse(
      """
    memory_store_config {
        extension {
          [type.googleapis.com/carls.memory_store.GaussianMemoryConfig] {
            per_cluster_buffer_size: %d
            distance_to_cluster_threshold: %f
            max_num_clusters: %d
            bootstrap_steps: %d
            min_variance: %f
            distance_type: CWISE_MEAN_GAUSSIAN
          }
        }
      }
  """ % (FLAGS.per_cluster_buffer_size, FLAGS.distance_to_cluster_threshold,
         FLAGS.max_num_clusters, FLAGS.bootstrap_steps, FLAGS.min_variance),
      config)
  return config


def build_model(kbs_address: typing.Text):
  """Constructs a ConvNet-based model configured from flags using keras API."""

  dm_config = build_dm_config_from_flags()
  mode = tf.constant(dm_ops.LOOKUP_WITH_UPDATE, dtype=tf.int32)

  # Builds normalization layer.
  if FLAGS.use_batch_normalization:
    normalization_layer = tf.keras.layers.BatchNormalization(
        axis=1, center=True, scale=True, momentum=0)
  else:
    normalization_layer = dn.DynamicNormalization(
        dm_config,
        mode=mode,
        axis=1,
        epsilon=0.001,
        service_address=kbs_address)

  # Number of target classes
  if FLAGS.dataset in ['mnist', 'cifar10']:
    num_classes = 10
  else:
    num_classes = 100

  # Builds a simple image model with a normalization layer at the top.
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(FLAGS.hidden_size, activation='relu'),
      normalization_layer,
      tf.keras.layers.Dense(num_classes, activation='softmax')
  ])
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
      metrics=['accuracy'],
  )

  return model


def train_model(kbs_address: typing.Text):
  """Train a model configured from flags using keras API."""

  # Loads and preprocesses data.
  ds_train, ds_test = tfds.load(FLAGS.dataset, split=['train', 'test'])
  ds_train = ds_train.batch(FLAGS.batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.map(
      preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.batch(4).prefetch(tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.map(
      preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  model = build_model(kbs_address)

  # Model training.
  model.fit(
      ds_train,
      epochs=FLAGS.num_epochs,
      validation_data=ds_test,
  )


def main(argv):
  del argv
  # Starts a local KBS server.
  options = kbs_server_helper.KnowledgeBankServiceOptions(True, -1, 10)
  kbs_server = kbs_server_helper.KbsServerHelper(options)
  kbs_address = 'localhost:%d' % kbs_server.port()
  train_model(kbs_address)


if __name__ == '__main__':
  app.run(main)
