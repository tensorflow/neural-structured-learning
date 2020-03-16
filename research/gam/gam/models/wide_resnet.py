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
"""A Resnet implementation from `Realistic Evaluation of Deep SSL Algorithms`.

Following this paper from Google Brain:
https://papers.nips.cc/paper/7585-realistic-evaluation-of-deep-semi-supervised-learning-algorithms
and this Github repository:
https://github.com/brain-research/realistic-ssl-evaluation
"""
from .models_base import Model

import numpy as np
import tensorflow as tf


def fast_flip(images, is_training):
  """Flips the input images when training."""

  def func(inp):
    batch_size = tf.shape(inp)[0]
    flips = tf.to_float(
        tf.random_uniform([batch_size, 1, 1, 1], 0, 2, tf.int32))
    flipped_inp = tf.reverse(inp, [2])
    return flips * flipped_inp + (1 - flips) * images

  return tf.cond(is_training, lambda: func(images), lambda: images)


class WideResnet(Model):
  """Resnet implementation from `Realistic Evaluation of Deep SSL Algorithms`.

  Attributes:
    num_classes: Integer representing the number of classes.
    lrelu_leakiness: A float representing the weight of the Leaky Relu
      parameter.
    horizontal_flip: A boolean specifying whether we do random horizontal flips
      of the training data, for data augmentation.
    random_translation: A boolean specifying whether we do random translations
      of the training data, for data augmentation.
    gaussian_noise: Boolean specifying whether to add Gaussian noise to the
      inputs.
    width: Integer representing the size of the convolutional filter.
    num_residual_units: Integer representing the number of residual units.
    name: String representing the model name.
    aggregation: String representing an aggregation operation that could be
      applied to the inputs. See superclass attributes for details.
    hidden_aggregation: A tuple or list of integers representing the number of
      units in each layer of aggregation multilayer percepron. After the inputs
      are passed through the encoding layers, before aggregation they are passed
      through a fully connected network with these numbers of hidden units in
      each layer.
    activation: An activation function to be applied to the outputs of each
      fully connected layer in the aggregation network.
    is_binary_classification: Boolean specifying if this is model for binary
      classification. If so, it uses a different loss function and returns
      predictions with a single dimension, batch size.
  """

  def __init__(self,
               num_classes,
               lrelu_leakiness,
               horizontal_flip,
               random_translation,
               gaussian_noise,
               width,
               num_residual_units,
               name="wide_resnet",
               ema_factor=None,
               is_binary_classification=False,
               aggregation=None,
               activation=tf.nn.leaky_relu,
               hidden_aggregation=()):
    super(WideResnet, self).__init__(
        aggregation=aggregation,
        hidden_aggregation=hidden_aggregation,
        activation=activation)
    self.name = name
    self.num_classes = num_classes
    self.is_binary_classification = is_binary_classification
    self.lrelu_leakiness = lrelu_leakiness
    self.horizontal_flip = horizontal_flip
    self.random_translation = random_translation
    self.gaussian_noise = gaussian_noise
    self.width = width
    self.num_residual_units = num_residual_units
    self.ema_factor = ema_factor

  def get_encoding_and_params(self,
                              inputs,
                              is_train,
                              update_batch_stats=True,
                              **unused_kwargs):
    """Creates the model hidden representations and prediction ops.

    For this model, the hidden representation is the last layer
    before the logit computation. The predictions are unnormalized logits.

    Args:
      inputs: A tensor containing the model inputs. The first dimension is the
        batch size.
      is_train: A placeholder representing a boolean value that specifies if
        this model will be used for training or for test.
      update_batch_stats: Boolean specifying whether to update the batch norm
        statistics.
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
      if isinstance(inputs, (list, tuple)):
        with tf.variable_scope("encoding"):
          left = self._get_encoding(inputs[0], is_train, update_batch_stats)
        with tf.variable_scope("encoding", reuse=True):
          right = self._get_encoding(inputs[1], is_train, update_batch_stats)
        encoding = self._aggregate((left, right))
      else:
        with tf.variable_scope("encoding"):
          encoding = self._get_encoding(inputs, is_train, update_batch_stats)

      # Store model variables for easy access.
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES,
          scope=tf.get_default_graph().get_name_scope())
      all_vars = {var.name: var for var in variables}

      reg_params = {}

    return encoding, all_vars, reg_params

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
      logits: A tensor of logits. For multiclass classification its
        shape is (num_samples, num_classes), where the second dimension contains
        a logit per class. For binary classification, its shape is
        (num_samples,), where each element is the probability of class 1 for
        that sample.
      all_vars: A dictionary mapping from variable name to TensorFlow op
        containing all variables used in this model.
      reg_params: A dictionary mapping from a variable name to a Tensor of
        parameters which will be used for regularization.
    """
    # Logits layer.
    with tf.variable_scope(self.name + "/prediction"):
      w_init = tf.glorot_normal_initializer()
      logits = tf.layers.dense(
          encoding, self.num_classes, kernel_initializer=w_init)
      if self.is_binary_classification:
        logits = logits[:, 0]

      # Store model variables for easy access.
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES,
          scope=tf.get_default_graph().get_name_scope())
      all_vars = {var.name: var for var in variables}

    # No regularization parameters.
    reg_params = {}

    return logits, all_vars, reg_params

  def _get_encoding(self, inputs, is_train, update_batch_stats, **kwargs):
    """Creates the model hidden representations and prediction ops.

    For this model, the hidden representation is the last layer before the logit
    computation. The predictions are unnormalized logits.

    Args:
      inputs: A tensor containing the model inputs. The first dimension is the
        batch size.
      is_train: A placeholder representing a boolean value that specifies if
        this model will be used for training or for test.
      update_batch_stats: Boolean specifying whether to update the batch norm
        statistics.
      **kwargs: Other keyword arguments.

    Returns:
      encoding: A tensor containing an encoded batch of samples. The first
        dimension corresponds to the batch size.
      all_vars: A dictionary mapping from variable name to TensorFlow op
        containing all variables used in this model.
      reg_params: A dictionary mapping from a variable name to a Tensor of
        parameters which will be used for regularization.
    """
    # Helper functions
    def _conv(name, x, filter_size, in_filters, out_filters, strides):
      """Convolution."""
      with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            "DW",
            [filter_size, filter_size, in_filters, out_filters],
            tf.float32,
            initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)),
        )
        return tf.nn.conv2d(x, kernel, strides, padding="SAME")

    def _relu(x, leakiness=0.0):
      """Relu, with optional leaky support."""
      return tf.where(tf.less(x, 0.0), leakiness * x, x, name="leaky_relu")

    def _residual(x,
                  in_filter,
                  out_filter,
                  stride,
                  activate_before_residual=False):
      """Residual unit with 2 sub layers."""
      if activate_before_residual:
        with tf.variable_scope("shared_activation"):
          x = tf.layers.batch_normalization(
              x, axis=1, scale=True, training=is_train)
          x = _relu(x, self.lrelu_leakiness)
          orig_x = x
      else:
        with tf.variable_scope("residual_only_activation"):
          orig_x = x
          x = tf.layers.batch_normalization(
              x, axis=1, scale=True, training=is_train)
          x = _relu(x, self.lrelu_leakiness)

      with tf.variable_scope("sub1"):
        x = _conv("conv1", x, 3, in_filter, out_filter, stride)

      with tf.variable_scope("sub2"):
        x = tf.layers.batch_normalization(
            x, axis=1, scale=True, training=is_train)
        x = _relu(x, self.lrelu_leakiness)
        x = _conv("conv2", x, 3, out_filter, out_filter, [1, 1, 1, 1])

      with tf.variable_scope("sub_add"):
        if in_filter != out_filter:
          orig_x = _conv("conv1x1", orig_x, 1, in_filter, out_filter, stride)
        x += orig_x
      return x

    x = inputs
    tf.summary.image("images_in_net", x)
    if self.horizontal_flip:
      x = fast_flip(x, is_training=is_train)
    if self.random_translation:
      raise NotImplementedError("Random translations are not implemented yet.")
    if self.gaussian_noise:
      x = tf.cond(is_train, lambda: x + tf.random_normal(tf.shape(x)) * 0.15,
                  lambda: x)
    x = _conv("init_conv", x, 3, 3, 16, [1, 1, 1, 1])

    activate_before_residual = [True, False, False]
    res_func = _residual
    filters = [16, 16 * self.width, 32 * self.width, 64 * self.width]

    with tf.variable_scope("unit_1_0"):
      x = res_func(x, filters[0], filters[1], [1, 1, 1, 1],
                   activate_before_residual[0])
    for i in range(1, self.num_residual_units):
      with tf.variable_scope("unit_1_%d" % i):
        x = res_func(x, filters[1], filters[1], [1, 1, 1, 1], False)

    with tf.variable_scope("unit_2_0"):
      x = res_func(x, filters[1], filters[2], [1, 2, 2, 1],
                   activate_before_residual[1])
    for i in range(1, self.num_residual_units):
      with tf.variable_scope("unit_2_%d" % i):
        x = res_func(x, filters[2], filters[2], [1, 1, 1, 1], False)

    with tf.variable_scope("unit_3_0"):
      x = res_func(x, filters[2], filters[3], [1, 2, 2, 1],
                   activate_before_residual[2])
    for i in range(1, self.num_residual_units):
      with tf.variable_scope("unit_3_%d" % i):
        x = res_func(x, filters[3], filters[3], [1, 1, 1, 1], False)

    with tf.variable_scope("unit_last"):
      x = tf.layers.batch_normalization(
          x, axis=1, scale=True, training=is_train)
      x = _relu(x, self.lrelu_leakiness)
      # Global average pooling.
      x = tf.reduce_mean(x, [1, 2])

    return x

  def get_loss(self,
               predictions,
               targets,
               name_scope="loss",
               reg_params=None,
               **kwargs):
    weight_decay = kwargs["weight_decay"] if "weight_decay" in kwargs else 0.0

    with tf.name_scope(name_scope):
      if self.is_binary_classification:
        loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets, logits=predictions))
      else:
        # Cross entropy error
        loss = tf.reduce_mean(
            tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=targets, logits=predictions),
                axis=-1))
      # Weight decay loss
      if reg_params:
        for var in reg_params.values():
          loss += weight_decay * tf.nn.l2_loss(var)
    return loss

  def normalize_predictions(self, predictions):
    if self.is_binary_classification:
      return tf.nn.sigmoid(predictions)
    return tf.nn.softmax(predictions, axis=-1)
