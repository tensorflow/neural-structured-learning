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
r"""An example Keras trainer for the Cora data set using graph regularization.

USAGE:
  python graph_keras_mlp_cora.py [flags] train.tfr test.tfr

See https://linqs.soe.ucsc.edu/data for a description of the Cora data set, and
the corresponding graph and training data set.

This example demonstrates the use of sequential, functional, and subclass models
in Keras for graph regularization. Users may change 'base_models' defined in
main() as necessary, to select a subset of the supported Keras base model types.
In all cases, the base model used is a multi-layer perceptron containing two
hidden layers with drop out.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
from absl import app
from absl import flags
from absl import logging
import attr

import neural_structured_learning as nsl
from research.carls import dynamic_embedding_config_pb2 as de_config_pb2
from research.carls import dynamic_embedding_neighbor_cache as de_nb_cache
from research.carls import graph_regularization
from research.carls import kbs_server_helper_pybind as kbs_server_helper
import tensorflow as tf
from google.protobuf import text_format


FLAGS = flags.FLAGS
FLAGS.showprefixforinfo = False

flags.DEFINE_integer('train_epochs', None, 'Number of epochs to train.')
flags.DEFINE_integer('eval_steps', None, 'Number of steps to evaluate.')
flags.DEFINE_string('output_dir', None, 'Output dir path for saved model.')

NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'


@attr.s
class HParams(object):
  """Hyper-parameters used for training."""
  ### dataset parameters
  num_classes = attr.ib(default=7)
  max_seq_length = attr.ib(default=1433)
  ### NGM parameters
  distance_type = attr.ib(default=nsl.configs.DistanceType.L2)
  graph_regularization_multiplier = attr.ib(default=0.1)
  num_neighbors = attr.ib(default=1)
  ### model architecture
  num_fc_units = attr.ib(default=[50, 50])
  ### training parameters
  train_epochs = attr.ib(default=10)
  batch_size = attr.ib(default=128)
  dropout_rate = attr.ib(default=0.5)
  ### eval parameters
  eval_steps = attr.ib(default=None)  # Every test instance is evaluated.


def get_hyper_parameters():
  """Returns the hyper-parameters used for training."""
  hparams = HParams()
  if FLAGS.train_epochs:
    hparams.train_epochs = FLAGS.train_epochs
  if FLAGS.eval_steps:
    hparams.eval_steps = FLAGS.eval_steps
  return hparams


def load_dataset(filename):
  """Reads a file in the `.tfrecord` format.

  Args:
    filename: Name of the file containing `tf.train.Example` objects.

  Returns:
    An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
    objects.
  """
  return tf.data.TFRecordDataset([filename])


def make_dataset(file_path, training, include_nbr_features, hparams):
  """Returns a `tf.data.Dataset` instance based on data in `file_path`."""

  def parse_example(example_proto):
    """Extracts relevant fields from the `example_proto`.

    Args:
      example_proto: An instance of `tf.train.Example`.

    Returns:
      A pair whose first value is a dictionary containing relevant features
      and whose second value contains the ground truth labels.
    """
    # The 'words' feature is a multi-hot, bag-of-words representation of the
    # original raw text. A default value is required for examples that don't
    # have the feature.
    feature_spec = {
        'id':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'words':
            tf.io.FixedLenFeature([hparams.max_seq_length],
                                  tf.int64,
                                  default_value=tf.constant(
                                      0,
                                      dtype=tf.int64,
                                      shape=[hparams.max_seq_length])),
        'label':
            tf.io.FixedLenFeature((), tf.int64, default_value=-1),
    }
    if include_nbr_features:
      for i in range(hparams.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'words')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i,
                                         NBR_WEIGHT_SUFFIX)
        nbr_id_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'id')
        feature_spec[nbr_feature_key] = tf.io.FixedLenFeature(
            [hparams.max_seq_length],
            tf.int64,
            default_value=tf.constant(
                0, dtype=tf.int64, shape=[hparams.max_seq_length]))
        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature(
            [1], tf.float32, default_value=tf.constant([0.0]))
        feature_spec[nbr_id_key] = tf.io.FixedLenFeature(
            (), tf.string, default_value='')

    features = tf.io.parse_single_example(example_proto, feature_spec)

    labels = features.pop('label')
    return features, labels

  # If the dataset is sharded, the following code may be required:
  # filenames = tf.data.Dataset.list_files(file_path, shuffle=True)
  # dataset = filenames.interleave(load_dataset, cycle_length=1)
  dataset = load_dataset(file_path)
  if training:
    dataset = dataset.shuffle(10000)
  dataset = dataset.map(parse_example)
  dataset = dataset.batch(hparams.batch_size)
  return dataset


def make_mlp_sequential_model(hparams):
  """Creates a sequential multi-layer perceptron model."""
  model = tf.keras.Sequential()
  model.add(
      tf.keras.layers.InputLayer(
          input_shape=(hparams.max_seq_length,), name='words'))
  # Input is already one-hot encoded in the integer format. We cast it to
  # floating point format here.
  model.add(
      tf.keras.layers.Lambda(lambda x: tf.keras.backend.cast(x, tf.float32)))
  for num_units in hparams.num_fc_units:
    model.add(tf.keras.layers.Dense(num_units, activation='relu'))
    model.add(tf.keras.layers.Dropout(hparams.dropout_rate))
  model.add(tf.keras.layers.Dense(hparams.num_classes, activation='softmax'))
  return model


def make_mlp_functional_model(hparams):
  """Creates a functional API-based multi-layer perceptron model."""
  inputs = tf.keras.Input(
      shape=(hparams.max_seq_length,), dtype='int64', name='words')

  # Input is already one-hot encoded in the integer format. We cast it to
  # floating point format here.
  cur_layer = tf.keras.layers.Lambda(
      lambda x: tf.keras.backend.cast(x, tf.float32))(
          inputs)

  for num_units in hparams.num_fc_units:
    cur_layer = tf.keras.layers.Dense(num_units, activation='relu')(cur_layer)
    # For functional models, by default, Keras ensures that the 'dropout' layer
    # is invoked only during training.
    cur_layer = tf.keras.layers.Dropout(hparams.dropout_rate)(cur_layer)

  outputs = tf.keras.layers.Dense(
      hparams.num_classes, activation='softmax')(
          cur_layer)

  model = tf.keras.Model(inputs, outputs=outputs)
  return model


def make_mlp_subclass_model(hparams):
  """Creates a multi-layer perceptron subclass model in Keras."""

  class MLP(tf.keras.Model):
    """Subclass model defining a multi-layer perceptron."""

    def __init__(self):
      super(MLP, self).__init__()
      self.cast_to_float_layer = tf.keras.layers.Lambda(
          lambda x: tf.keras.backend.cast(x, tf.float32))
      self.dense_layers = [
          tf.keras.layers.Dense(num_units, activation='relu')
          for num_units in hparams.num_fc_units
      ]
      self.dropout_layer = tf.keras.layers.Dropout(hparams.dropout_rate)
      self.output_layer = tf.keras.layers.Dense(
          hparams.num_classes, activation='softmax')

    def call(self, inputs, training=False):
      cur_layer = self.cast_to_float_layer(inputs['words'])
      for dense_layer in self.dense_layers:
        cur_layer = dense_layer(cur_layer)
        cur_layer = self.dropout_layer(cur_layer, training=training)

      outputs = self.output_layer(cur_layer)

      return outputs

  return MLP()


def log_metrics(model_desc, eval_metrics):
  """Logs evaluation metrics at `logging.INFO` level.

  Args:
    model_desc: A description of the model.
    eval_metrics: A dictionary mapping metric names to corresponding values. It
      must contain the loss and accuracy metrics.
  """
  logging.info('\n')
  logging.info('Eval accuracy for %s: %s', model_desc, eval_metrics['accuracy'])
  logging.info('Eval loss for %s: %s', model_desc, eval_metrics['loss'])
  if 'graph_loss' in eval_metrics:
    logging.info('Eval graph loss for %s: %s', model_desc,
                 eval_metrics['graph_loss'])


def train_and_evaluate(model, model_desc, model_path, train_dataset,
                       test_dataset, hparams):
  """Compiles, trains, and evaluates a `Keras` model.

  Args:
    model: An instance of `tf.Keras.Model`.
    model_desc: A description of the model.
    model_path: The path to save the model.
    train_dataset: An instance of `tf.data.Dataset` representing training data.
    test_dataset: An instance of `tf.data.Dataset` representing test data.
    hparams: An instance of `Hparams`.
  """
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
      metrics=['accuracy'])
  callbacks = []
  if model_path:
    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        model_path + '/weights-{epoch:02d}',
        save_weights_only=True,
        save_freq='epoch'
    )]
  model.fit(
      train_dataset,
      epochs=hparams.train_epochs,
      verbose=1,
      callbacks=callbacks)
  eval_results = dict(
      zip(model.metrics_names,
          model.evaluate(test_dataset, steps=hparams.eval_steps)))
  log_metrics(model_desc, eval_results)


def kbs_config(embedding_dimension):
  """Returns the configuration to use dynamic embedding neighbor cache."""
  config = de_config_pb2.DynamicEmbeddingConfig()
  text_format.Parse(
      """
    embedding_dimension: %d
    knowledge_bank_config {
      initializer {
        random_uniform_initializer {
          low: -0.5
          high: 0.5
        }
        use_deterministic_seed: true
      }
      extension {
        [type.googleapis.com/carls.InProtoKnowledgeBankConfig] {}
      }
    }
  """ % embedding_dimension, config)
  return config


def start_server():
  options = kbs_server_helper.KnowledgeBankServiceOptions(True, -1, 10)
  server = kbs_server_helper.KbsServerHelper(options)
  kbs_address = 'localhost:%d' % server.port()
  return server, kbs_address


def build_model(base_model_tag, hparams):
  if base_model_tag == 'FUNCTIONAL':
    model = make_mlp_functional_model(hparams)
  elif base_model_tag == 'SEQUENTIAL':
    model = make_mlp_sequential_model(hparams)
  else:
    model = make_mlp_subclass_model(hparams)
  return model


def update_embeddings(
    stop_event, model_path, model, neighbor_cache_client, dataset):
  """Updates embeddings in knowledge bank server.

  This runs in a background thread, loads the latest unused checkpoint,
  then runs inference on all available data, updates corresponding embeddings
  in the knowledge bank server.

  Args:
    stop_event: `threading.Event` object used to stop updating the embeddings.
    model_path: path to the saved model.
    model: `GraphRegularizationWithCaching` object for inference.
    neighbor_cache_client: `NeighborCacheClient` object to update embeddings.
    dataset: dataset for inference to update.
  """
  logging.info('Start embedding updates')
  last_used_ckpt_path = None

  def update_embedding_fn(features, label):
    if not stop_event.wait(0.01):
      neighbor_cache_client.update(
          features['id'], model.base_model(features))
    return features, label

  # Keep updating until stop_event set to True.
  while not stop_event.wait(0.1):
    # Keep waiting to load the latest checkpoint until it exists.
    while not stop_event.wait(0.1):
      try:
        latest_ckpt_path = tf.train.latest_checkpoint(model_path)
        if last_used_ckpt_path != latest_ckpt_path:
          # No new checkpoint since the last update.
          last_used_ckpt_path = latest_ckpt_path
          model.load_weights(latest_ckpt_path)
          break
      except tf.errors.NotFoundError:
        pass

    # Run inference on the dataset and update embeddings.
    dataset.map(update_embedding_fn)

  logging.info('Finished embedding updates')


def main(argv):
  # Check that the correct number of arguments have been provided. The
  # training and test data should contain 'tf.train.Example' objects in the
  # TFRecord format.
  if len(argv) != 3:
    raise app.UsageError('Invalid number of arguments; expected 2, got %d' %
                         (len(argv) - 1))
  if not FLAGS.output_dir:
    raise app.UsageError('Must specify the output dir')

  hparams = get_hyper_parameters()
  train_data_path = argv[1]
  test_data_path = argv[2]

  # Graph regularization configuration.
  graph_reg_config = nsl.configs.make_graph_reg_config(
      max_neighbors=hparams.num_neighbors,
      multiplier=hparams.graph_regularization_multiplier,
      distance_type=hparams.distance_type,
      sum_over_axis=-1)

  # Create the base MLP models.
  base_model_tags = ['FUNCTIONAL', 'SEQUENTIAL', 'SUBCLASS']
  for base_model_tag in base_model_tags:
    base_model = build_model(base_model_tag, hparams)
    logging.info('\n====== %s BASE MODEL TEST BEGIN ======', base_model_tag)
    train_dataset = make_dataset(train_data_path, True, False, hparams)
    test_dataset = make_dataset(test_data_path, False, False, hparams)
    train_and_evaluate(base_model, 'Base MLP model', None, train_dataset,
                       test_dataset, hparams)

    logging.info('\n====== TRAINING WITH GRAPH REGULARIZATION ======\n')
    kbs_server, kbs_address = start_server()
    config = kbs_config(hparams.num_classes)
    client = de_nb_cache.DynamicEmbeddingNeighborCache(
        'id', config, kbs_address, timeout_ms=10 * 1000)
    graph_reg_model = graph_regularization.GraphRegularizationWithCaching(
        base_model, graph_reg_config, client)

    train_dataset = make_dataset(train_data_path, True, True, hparams)
    test_dataset = make_dataset(test_data_path, False, False, hparams)
    graph_model_path = os.path.join(
        FLAGS.output_dir, 'model_' + base_model_tag)

    # Start a background thread to update emebddings.
    stop_event = threading.Event()
    base_inference_model = build_model(base_model_tag, hparams)
    graph_inference_model = graph_regularization.GraphRegularizationWithCaching(
        base_inference_model, graph_reg_config, client)
    update_thread = threading.Thread(
        target=update_embeddings,
        args=(stop_event, graph_model_path,
              graph_inference_model, client, train_dataset))
    update_thread.daemon = True
    update_thread.start()

    # Train graph-regularized model using historical neighbor embeddings.
    train_and_evaluate(graph_reg_model, 'MLP + graph regularization',
                       graph_model_path, train_dataset, test_dataset, hparams)
    stop_event.set()
    kbs_server.Terminate()
    update_thread.join()

    logging.info('\n====== %s BASE MODEL TEST END ======', base_model_tag)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  app.run(main)
