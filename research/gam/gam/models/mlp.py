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
"""Implementation of a Multilayer Perceptron for classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .models_base import glorot
from .models_base import Model

import numpy as np
import tensorflow as tf


class MLP(Model):
  """Multilayer Perceptron for binary and multi-class classification.

  Attributes:
    output_dim: Integer representing the number of classes.
    hidden_sizes: List containing the sizes of the hidden layers.
    activation: An activation function to apply to the output of each hidden
      layer.
    aggregation: String representing an aggregation operation that could be
      applied to the inputs. Valid options: None, `add`. If None, then no
      aggregation is performed. If `add`, the first half of the features
      dimension is added to the second half (see the `_aggregate` function
      for details).
    hidden_aggregation: A tuple or list of integers representing the number of
      hidden units in each layer of the projection network described above.
    is_binary_classification: Boolean specifying if this is model for
      binary classification. If so, it uses a different loss function and
      returns predictions with a single dimension, batch size.
    name: String representing the model name.
  """

  def __init__(self,
               output_dim,
               hidden_sizes,
               activation=tf.nn.leaky_relu,
               aggregation=None,
               hidden_aggregation=(),
               is_binary_classification=False,
               name='MLP'):
    super(MLP, self).__init__(
        aggregation=aggregation,
        hidden_aggregation=hidden_aggregation,
        activation=activation)
    self.output_dim = output_dim
    self.hidden_sizes = hidden_sizes
    self.is_binary_classification = is_binary_classification
    self.name = name

  def _construct_layers(self, inputs):
    """Creates all hidden layers of the model, before the prediction layer.

    Args:
      inputs: A tensor containing the model inputs. The first dimension is the
        batch size.

    Returns:
      A tuple containing the encoded representation of the inputs and a
      dictionary of regularization parameters.
    """
    reg_params = {}
    # Reshape inputs in case they are not of shape (batch_size, features).
    num_features = np.prod(inputs.shape[1:])
    inputs = tf.reshape(inputs, [-1, num_features])
    hidden = inputs
    for layer_index, output_size in enumerate(self.hidden_sizes):
      input_size = hidden.get_shape().dims[-1].value
      weights_name = 'W_' + str(layer_index)
      weights = tf.get_variable(
          name=weights_name,
          initializer=glorot((input_size, output_size)),
          use_resource=True)
      reg_params[weights_name] = weights
      biases = tf.get_variable(
          'b_' + str(layer_index),
          initializer=tf.zeros([output_size], dtype=tf.float32),
          use_resource=True)
      hidden = self.activation(tf.nn.xw_plus_b(hidden, weights, biases))
    return hidden, reg_params

  def get_encoding_and_params(self, inputs, **unused_kwargs):
    """Creates the model hidden representations and prediction ops.

    For this model, the hidden representation is the last layer of the MLP,
    before the logit computation. The predictions are unnormalized logits.

    Args:
      inputs: A tensor containing the model inputs. The first dimension is the
        batch size.
      **unused_kwargs: Other unused keyword arguments.

    Returns:
      encoding: A tensor containing an encoded batch of samples. The first
        dimension corresponds to the batch size.
      all_vars: A dictionary mapping from variable name to TensorFlow op
        containing all variables used in this model.
      reg_params: A dictionary mapping from a variable name to a Tensor of
        parameters which will be used for regularization.
    """
    # Build layers.
    with tf.variable_scope(self.name):
      if isinstance(inputs, (tuple, list)):
        with tf.variable_scope('encoding'):
          hidden1, reg_params = self._construct_layers(inputs[0])
        with tf.variable_scope('encoding', reuse=True):
          hidden2, _ = self._construct_layers(inputs[1])
        hidden = self._aggregate((hidden1, hidden2))
      else:
        with tf.variable_scope('encoding'):
          hidden, reg_params = self._construct_layers(inputs)

      # Store model variables for easy access.
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES,
          scope=tf.get_default_graph().get_name_scope())
      all_vars = {var.name: var for var in variables}

    return hidden, all_vars, reg_params

  def get_predictions_and_params(self, encoding, is_train, **kwargs):
    """Creates the model prediction op.

    For this model, the hidden representation is the last layer of the MLP,
    before the logit computation. The predictions are unnormalized logits.

    Args:
      encoding: A tensor containing the model inputs. The first dimension is the
        batch size.
      is_train: A placeholder representing a boolean value that specifies if
        this model will be used for training or for test.
      **kwargs: Other keyword arguments.

    Returns:
      predictions: A tensor of logits. For multiclass classification its
        shape is (num_samples, num_classes), where the second dimension contains
        a logit per class. For binary classification, its shape is
        (num_samples,), where each element is the probability of class 1 for
        that sample.
      all_vars: A dictionary mapping from variable name to TensorFlow op
        containing all variables used in this model.
      reg_params: A dictionary mapping from a variable name to a Tensor of
        parameters which will be used for regularization.
    """
    reg_params = {}

    # Build layers.
    with tf.variable_scope(self.name + '/prediction'):
      input_size = encoding.get_shape().dims[-1].value
      weights = tf.get_variable(
          'W_outputs',
          initializer=glorot((input_size, self.output_dim)),
          use_resource=True)
      reg_params['W_outputs'] = weights
      biases = tf.get_variable(
          'b_outputs',
          initializer=tf.zeros([self.output_dim], dtype=tf.float32),
          use_resource=True)
      predictions = tf.nn.xw_plus_b(encoding, weights, biases,
                                    name='predictions')
      if self.is_binary_classification:
        predictions = predictions[:, 0]

      # Store model variables for easy access.
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES,
          scope=tf.get_default_graph().get_name_scope())
      all_vars = {var.name: var for var in variables}

    return predictions, all_vars, reg_params

  def get_loss(self,
               predictions,
               targets,
               name_scope='loss',
               reg_params=None,
               **kwargs):
    """Returns a loss between the provided targets and predictions.

    For binary classification, this loss is sigmoid cross entropy. For
    multi-class classification, it is softmax cross entropy.
    A weight decay loss is also added to the parameters passed in reg_params.

    Args:
      predictions: A tensor of predictions. For multiclass classification its
        shape is (num_samples, num_classes), where the second dimension contains
        a logit per class. For binary classification, its shape is
        (num_samples,), where each element is the probability of class 1 for
        that sample.
      targets: A tensor of targets of shape (num_samples,), where each row
        contains the label index of the corresponding sample.
      name_scope: A string containing the name scope used in TensorFlow.
      reg_params: A dictonary of parameters, mapping from name to parameter, for
        the variables to be included in the weight decay loss. If None, no
        weight decay is applied.
      **kwargs: Keyword arguments, potentially containing the weight of the
        regularization term, passed under the name `weight_decay`. If this is
        not provided, it defaults to 0.0.

    Returns:
      loss: The cummulated loss value.
    """
    reg_params = reg_params if reg_params is not None else {}
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else None

    with tf.name_scope(name_scope):
      # Cross entropy error.
      if self.is_binary_classification:
        loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets, logits=predictions))
      else:
        loss = tf.losses.softmax_cross_entropy(targets, predictions)
      # Weight decay loss.
      if weight_decay is not None:
        for var in reg_params.values():
          loss = loss + weight_decay * tf.nn.l2_loss(var)
    return loss

  def normalize_predictions(self, predictions):
    """Converts predictions to probabilities.

    Args:
      predictions: A tensor of logits. For multiclass classification its shape
        is (num_samples, num_classes), where the second dimension contains a
        logit per class. For binary classification, its shape is (num_samples,),
        where each element is the probability of class 1 for that sample.

    Returns:
      A tensor of the same shape as predictions, with values between [0, 1]
    representing probabilities.
    """
    if self.is_binary_classification:
      return tf.nn.sigmoid(predictions)
    return tf.nn.softmax(predictions, axis=-1)
