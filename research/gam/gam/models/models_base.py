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
"""Code used by all models use by Graph Agreement Models."""
import abc
import logging

import numpy as np
import tensorflow as tf


def glorot(shape, name=None):
  """Glorot & Bengio (AISTATS 2010) initialization."""
  init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
  initial = tf.random_uniform(
      shape, minval=-init_range, maxval=init_range, dtype=tf.float32, name=name)
  return initial


class Model(object):
  """Superclass for models used for the classification or agreement model.

  Attributes:
    aggregation: A string representing the way to aggregate two inputs provided
      in the function `_aggregate`. This is designed for the agreement model,
      where the source and target inputs need to be integrated. The allowed
      options are the following:
        None: no aggregation is performed.
        - `add`: the two inputs are added
        - `dist`: squared distance between the two inputs, elementwise.
        - `concat`: the two inputs are concatenated along axis=1.
        - `project_add`: the two inputs are first projected to a different space
          using a fully connected network with the hidden units provided in
          hidden_aggregation, and followed by the activation function in
          activation. Then the projected inputs are added, like in `add`.
        - `project_dist`: The two inputs are projected as in `project_add`,
          followed by a distance calculation like in `dist`.
        - `project_concat`: The two inputs are projected as in `project_add`,
          followed by a concatenation like in `concat`.
    hidden_aggregation: A tuple or list of integers representing the number of
      hidden units in each layer of the projection network described above.
    activation: An activation function to be applied to the outputs of each
      fully connected layer of the aggregation multilayer perceptron.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, aggregation=None, hidden_aggregation=(),
               activation=lambda x: x):
    self.aggregation = aggregation
    self.hidden_aggregation = hidden_aggregation
    self.activation = activation

    assert aggregation in (None, 'add', 'dist', 'concat', 'project_add',
                           'project_dist', 'project_concat')

  @abc.abstractmethod
  def get_predictions_and_params(self, inputs, is_train, **kwargs):
    """Creates model variables and returns the predictions and parameters."""
    pass

  @abc.abstractmethod
  def get_loss(self, predictions, targets, reg_params, **kwargs):
    """Returns the loss op to be optimized in order to train this model."""
    pass

  @abc.abstractmethod
  def normalize_predictions(self, predictions):
    """Convert predictions to probabilities."""
    pass

  def __call__(self, inputs, **kwargs):
    return self.get_predictions_and_params(inputs, **kwargs)

  def save(self, variables, path, session):
    """Saves a model using a Tensorflow Saver."""
    saver = tf.train.Saver(variables)
    save_path = saver.save(session, path)
    logging.info('Model saved in file: %s', save_path)

  def load(self, variables, path, session):
    """Loads a model using a Tensorflow Saver."""
    saver = tf.train.Saver(variables)
    saver.restore(session, path)
    logging.info('Model restored from file: %s', path)

  def _aggregate(self, inputs):
    """Aggregates the input features.

    Because this MLP can be used both by the classification and agreement
    models, the provided inputs could be a single batch of features for the
    classification model, or a tuple of two batches (src_features, tgt_features)
    for the agreement model. In the latter case, the two inputs need to be
    aggregated before they are passed through the MLP. Here we provide several
    options for how this aggregation can be done, specified by the `aggregate`
    class attribute. The valid options are:
    - None: no aggregation is performed (this is used by the classification
            model).
    - add: The two inputs are added: src_features + tgt_features.
    - dist: The two inputs are aggregated into their squared difference.
    - concat: The two inputs are concatenated along the features dimension.
    - project_add: The two inputs are projected to another space, and then
                   added.
    - project_dist: The two inputs are projected to another space, and then
                    we compute squared element-wise distance.
    - project_concat: The two inputs are projected to another space, and then
                      concatenated.

    Args:
      inputs: A batch of features of shape (batch_size, num_features) or
        a tuple of two such batches of features.
    Returns:
      A batch of aggregated features of shape (batch_size, new_num_features),
      where the feature dimension size may have changed.
    """
    if self.aggregation is None:
      return inputs
    # If it requires aggregation, we assume the inputs are passes as a tuple.
    left = inputs[0]
    right = inputs[1]

    # In case the aggregation option requires projection, we first do this.
    if self.aggregation in ('project_add', 'project_dist', 'project_concat'):
      left = self._project(left)
      right = self._project(right, reuse=True)

    if self.aggregation.endswith('add'):
      return left + right
    if self.aggregation.endswith('dist'):
      return tf.square(left - right)
    elif self.aggregation.endswith('concat'):
      return tf.concat((left, right), axis=1)
    else:
      raise NotImplementedError()

  def _project(self, inputs, reuse=tf.compat.v1.AUTO_REUSE):
    """Projects the provided inputs using a multilayer perceptron.

    Args:
      inputs: A batch of features whose first dimension is the batch size.
      reuse: A boolean specifying whether to reuse the same projection weights
        or create new ones.

    Returns:
      Projected inputs, having the same batch size as the first dimension.
    """
    with tf.variable_scope('aggregation', reuse=reuse):
      # Reshape inputs in case they have more than 2 dimensions.
      num_features = inputs.get_shape().dims[1:]
      num_features = np.prod([f.value for f in num_features])
      if len(inputs.shape) > 2:
        inputs = tf.reshape(inputs, [-1, num_features])
      # Create the fully connected layers.
      hidden = inputs
      for layer_index, num_units in enumerate(self.hidden_aggregation):
        input_size = hidden.get_shape().dims[-1].value
        weights = tf.get_variable(
            'W_' + str(layer_index),
            initializer=glorot((input_size, num_units)),
            use_resource=True)
        bias = tf.get_variable(
            'b_' + str(layer_index),
            shape=(num_units,),
            initializer=tf.zeros_initializer(),
            use_resource=True)
        hidden = self.activation(tf.nn.xw_plus_b(hidden, weights, bias))
      return hidden
