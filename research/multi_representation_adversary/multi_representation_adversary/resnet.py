# Copyright 2020 Google LLC
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

"""ResNet V1 implementation.

Derived from:
  https://github.com/google-research/google-research/blob/master/uq_benchmark_2019/cifar/resnet.py
"""

from absl import logging
import gin
import tensorflow.compat.v2 as tf


def _resnet_layer(inputs,
                  num_filters=16,
                  kernel_size=3,
                  strides=1,
                  activation="relu",
                  batch_norm=True,
                  conv_first=True,
                  l2_regularization_weight=1e-4):
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs (tensor): input tensor from input image or previous layer
    num_filters (int): Conv2D number of filters
    kernel_size (int): Conv2D square kernel dimensions
    strides (int): Conv2D square stride dimensions
    activation (string): Activation function string.
    batch_norm (bool): whether to include batch normalization
    conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)
    l2_regularization_weight: The parameter for the L2 regularizer.

  Returns:
      x (tensor): tensor as input to the next layer
  """
  conv = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding="same",
      kernel_initializer="he_normal",
      kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_weight))

  if batch_norm:
    batch_norm_fn = tf.keras.layers.BatchNormalization()
  else:
    batch_norm_fn = lambda x: x
  if activation is not None:
    activation_fn = tf.keras.layers.Activation(activation)
  else:
    activation_fn = lambda x: x

  x = inputs
  if conv_first:
    x = conv(x)
    x = batch_norm_fn(x)
    x = activation_fn(x)
  else:
    x = batch_norm_fn(x)
    x = activation_fn(x)
    x = conv(x)
  return x


################################################################################


@gin.configurable
def build_resnet_v1(input_shape,
                    depth,
                    num_classes=10,
                    num_filters=16,
                    l2_reg_weight=1e-4,
                    return_logits=False):
  """Returns a Model object for ResNet V1."""
  if (depth - 2) % 6 != 0:
    raise ValueError("depth should be 6n+2 (eg 14, 20, 32, 44, 50)")
  # Start model definition.
  num_res_blocks = int((depth - 2) / 6)
  logging.info("Starting ResNet build.")
  inputs = tf.keras.layers.Input(shape=input_shape, name="image")

  x = _resnet_layer(
      inputs, num_filters=num_filters, l2_regularization_weight=l2_reg_weight)
  # Instantiate the stack of residual units
  for stack in range(3):
    for res_block in range(num_res_blocks):
      logging.info("Starting ResNet stack #%d block #%d.", stack, res_block)
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = _resnet_layer(
          inputs=x,
          num_filters=num_filters,
          strides=strides,
          l2_regularization_weight=l2_reg_weight)
      y = _resnet_layer(
          inputs=y,
          num_filters=num_filters,
          activation=None,
          l2_regularization_weight=l2_reg_weight)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match changed dims
        x = _resnet_layer(
            inputs=x,
            num_filters=num_filters,
            kernel_size=1,
            strides=strides,
            activation=None,
            batch_norm=False,
            l2_regularization_weight=l2_reg_weight)
      x = tf.keras.layers.add([x, y])
      x = tf.keras.layers.Activation("relu")(x)
    num_filters *= 2

  # Add classifier on top.
  # v1 does not use BN after last shortcut connection-ReLU
  x = tf.keras.layers.AveragePooling2D(pool_size=x.shape[1:3])(x)
  y = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(
      num_classes,
      activation=None if return_logits else "softmax",
      kernel_initializer="he_normal")(
          y)

  return tf.keras.Model(inputs=inputs, outputs=outputs)
