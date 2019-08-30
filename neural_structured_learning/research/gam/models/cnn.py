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
"""A convolutional neural network architecture for image classification.

This architecture is used in the TensorFlow tutorial for CIFAR10:
https://www.tensorflow.org/tutorials/images/deep_cnn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gam.models import Model
import numpy as np
import tensorflow as tf


class ImageCNNAgreement(Model):
  """Convolutional Neural Network for image classification.

  It assumes the inputs are images of shape width x height x channels.
  The precise architecture follows the Tensorflow CNN
  tutorial at https://www.tensorflow.org/tutorials/images/deep_cnn.
  Note that this CNN works both for the agreement and classification models.
  In the agreement case, the provided inputs will be a tuple
  (inputs_src, inputs_target), which are aggregated into one input after the
  convolution layers, right before the fully connected network that makes the
  final prediction.

  Attributes:
    output_dim: Integer representing the number of classes.
    channels: Integer representing the number of channels in the input images
      (e.g., 1 for black and white, 3 for RGB).
    aggregation: String representing an aggregation operation that could be
      applied to the inputs. See superclass attributes for details.
    hidden_prediction: A tuple or list of integers representing the number of
      units in each layer of output multilayer percepron. After the inputs are
      passed through the convolution layers (and potentially aggregated), they
      are passed through a fully connected network with these numbers of hidden
      units in each layer.
    activation: An activation function to be applied to the outputs of each
      fully connected layer.
    is_binary_classification: Boolean specifying if this is model for
      binary classification. If so, it uses a different loss function and
      returns predictions with a single dimension, batch size.
    name: String representing the model name.
  """

  def __init__(self,
               output_dim,
               channels,
               hidden_prediction=(384, 192),
               aggregation=None,
               hidden_aggregation=(),
               activation=tf.nn.leaky_relu,
               is_binary_classification=False,
               name='cnn_agr'):
    super(ImageCNNAgreement, self).__init__(
        aggregation=aggregation,
        hidden_aggregation=hidden_aggregation,
        activation=activation)
    self.output_dim = output_dim
    self.channels = channels
    self.hidden_prediction = hidden_prediction
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
    # Convolution 1.
    with tf.variable_scope('conv1') as scope:
      kernel = tf.get_variable(
          'kernel',
          shape=[5, 5, self.channels, 64],
          initializer=tf.truncated_normal_initializer(stddev=5e-2,
                                                      dtype=tf.float32),
          dtype=tf.float32)
      conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.get_variable(
          'biases',
          [64],
          initializer=tf.constant_initializer(0.0),
          dtype=tf.float32)
      pre_activation = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.leaky_relu(pre_activation, name=scope.name)

    # Max pooling 1.
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # Local Response Normalization 1.
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    # Convolution 2.
    with tf.variable_scope('conv2') as scope:
      kernel = tf.get_variable(
          'kernel',
          shape=[5, 5, 64, 64],
          initializer=tf.truncated_normal_initializer(stddev=5e-2,
                                                      dtype=tf.float32),
          dtype=tf.float32)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.get_variable(
          'biases',
          [64],
          initializer=tf.constant_initializer(0.1),
          dtype=tf.float32)
      pre_activation = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.leaky_relu(pre_activation, name=scope.name)

    # Local Response Normalization 2.
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # Max pooling 2.
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    shape = pool2.get_shape().as_list()
    dim = np.prod(shape[1:])
    encoding = tf.reshape(pool2, [-1, dim])

    # There are no regularization parameters for this model, but for interface
    # consistency with other models, we return an empty dictionary.
    reg_params = {}

    return encoding, reg_params

  def get_encoding_and_params(self, inputs, is_train, **kwargs):
    """Creates the model hidden representations and prediction ops.

    For this model, the hidden representation is the last layer
    before the logit computation. The predictions are unnormalized logits.

    Args:
      inputs: A tensor containing the model inputs. The first dimension is the
        batch size.
      is_train: A placeholder representing a boolean value that specifies if
        this model will be used for training or for test.
      **kwargs: Other keyword arguments.

    Returns:
      encoding: A tensor containing an encoded batch of samples. The first
        dimension corresponds to the batch size.
      all_vars: A dictionary mapping from variable name to TensorFlow op
        containing all variables used in this model.
      reg_params: A dictionary mapping from a variable name to a Tensor of
        parameters which will be used for regularization.
    """
    # Build layers.
    scope = self.name + '/encoding'
    with tf.variable_scope(scope):
      if isinstance(inputs, (list, tuple)):
        # If we have multiple inputs (e.g., in the case of the agreement model),
        # split into left and right inputs, compute the hidden representation of
        # each branch, then aggregate.
        left = inputs[0]
        right = inputs[1]
        with tf.variable_scope('hidden'):
          hidden1, reg_params = self._construct_layers(left)
        with tf.variable_scope('hidden', reuse=True):
          hidden2, _ = self._construct_layers(right)
        hidden = self._aggregate((hidden1, hidden2))
      else:
        with tf.variable_scope('hidden'):
          hidden, reg_params = self._construct_layers(inputs)

      # Store model variables for easy access.
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES,
          scope=tf.get_default_graph().get_name_scope())
      all_vars = {var.name: var for var in variables}

    return hidden, all_vars, reg_params

  def _construct_prediction(self, inputs):
    """Creates the last layer of the model and returns its predictions."""
    with tf.variable_scope('prediction'):
      # We store all variables on which we apply weight decay in a dictionary.
      reg_params = {}
      # Create the hidden layers of the prediction MLP.
      with tf.variable_scope('hidden'):
        hidden = inputs
        for layer_index, output_size in enumerate(self.hidden_prediction):
          input_size = hidden.get_shape().dims[-1].value
          weights_name = 'W_' + str(layer_index)
          weights = tf.get_variable(
              name=weights_name,
              shape=(input_size, output_size),
              initializer=tf.truncated_normal_initializer(stddev=0.04,
                                                          dtype=tf.float32),
              use_resource=True)
          reg_params[weights_name] = weights
          biases = tf.get_variable(
              'b_' + str(layer_index),
              initializer=tf.zeros([output_size], dtype=tf.float32),
              use_resource=True)
          hidden = self.activation(tf.nn.xw_plus_b(hidden, weights, biases))
      # Create the output layer of the predictions.
      with tf.variable_scope('output'):
        input_size = hidden.get_shape().dims[-1].value
        weights = tf.get_variable(
            'W_outputs',
            shape=(input_size, self.output_dim),
            initializer=tf.truncated_normal_initializer(stddev=1.0/input_size,
                                                        dtype=tf.float32),
            use_resource=True)
        biases = tf.get_variable(
            'b_outputs',
            initializer=tf.zeros([self.output_dim], dtype=tf.float32),
            use_resource=True)
        predictions = tf.nn.xw_plus_b(
            hidden, weights, biases, name='predictions')
      if self.is_binary_classification:
        predictions = predictions[:, 0]
    return predictions, reg_params

  def get_predictions_and_params(self, encoding, is_train, **kwargs):
    """Creates the model prediction op.

    For this model, the hidden representation is the last layer
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
    # Build layers.
    with tf.variable_scope(self.name + '/prediction'):
      predictions, reg_params = self._construct_prediction(encoding)

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

    Arguments:
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
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0.0

    with tf.name_scope(name_scope):
      # Cross entropy error.
      if self.is_binary_classification:
        loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets, logits=predictions))
      else:
        loss = tf.losses.softmax_cross_entropy(targets, predictions)
      # Weight decay loss.
      for var in reg_params.values():
        loss += weight_decay * tf.nn.l2_loss(var)
    return loss

  def normalize_predictions(self, predictions):
    """Converts predictions to probabilities.

    Arguments:
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
