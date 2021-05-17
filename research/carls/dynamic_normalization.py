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
"""Implementation of DynamicNormalization.

DynamicNormalization differs from BatchNormalization in the following aspects:

1) It assumes each input activation belongs to one of many clusters and the
   number of clusters can grow dynamically in mode dm_ops.LOOKUP_WITH_GROW.

2) The normalization equation is derived based on the assumption that a layer
   computes a Gaussian distribution, and it is shown that the resulting models
   often outperform BatchNormalization.

3) Compared to BatchNormalization, DynamicNormalization works well in any batch
   size.
"""

import typing

from research.carls import dynamic_embedding_config_pb2 as de_config_pb2
from research.carls import dynamic_memory_ops as dm_ops
import tensorflow as tf


class DynamicNormalization(tf.keras.layers.Layer):
  r"""Keras' layer implementation for DynamicNormalization.

  Similar to Batch Normalization, DynamicNormalization normalizes the
  activations of the previous layer for each input.

  The normalization formula:
    ((|mean|^2 + p(mean)) / |x|^2 - 2 * g(mean)) / sqrt(variance)
  where
    p(mean) = prior_scale * mean + prior_offset
    g(mean) = mean_scale * mean + mean_offset.
  Here (prior_scale, prior_offset, mean_scale, mean_offset) are learnable
  parameters.
  """

  def __init__(self,
               dm_config: de_config_pb2.DynamicEmbeddingConfig,
               mode: int,
               axis: int = -1,
               epsilon: float = 1e-3,
               scale_initializer='ones',
               offset_initializer='zeros',
               scale_regularizer=None,
               offset_regularizer=None,
               scale_constraint=None,
               offset_constraint=None,
               use_batch_normalization: bool = False,
               trainable=True,
               service_address: typing.Text = '',
               name=None,
               **kwargs):
    r"""Constructor of DynamicNormalization.

    Args:
      dm_config: An instance of DynamicEmbeddingConfig.
      mode: An int or a `Tensor` whose value must be  one of
        {LOOKUP_WITHOUT_UPDATE, LOOKUP_WITH_UPDATE, LOOKUP_WITH_GROW} as defined
        in dynamic_memory_ops.py.
      axis: Integer, the axis along which to compute mean and variance.
      epsilon: Small float added to variance to avoid dividing by zero.
      scale_initializer: Initializer for the scale weight.
      offset_initializer: Initializer for the offset weight.
      scale_regularizer: Optional regularizer for the scale weight.
      offset_regularizer: Optional regularizer for the offset weight.
      scale_constraint: Optional constraint for the scale weight.
      offset_constraint: Optional constraint for the offset weight.
      use_batch_normalization: Boolean, if `True`, use BatchNormalization's
        formula instead of DynamicNormalization's own one when computing the
        output.
      trainable: Boolean, if `True` the variables will be marked as trainable.
      service_address: The address of a knowledge bank service. If empty, the
        value passed from --kbs_address flag will be used instead.
      name: A string indicating the op's name.
      **kwargs: Addition inputs.
    """
    super(DynamicNormalization, self).__init__(name=name, **kwargs)
    if mode is None:
      raise ValueError('Must specify model mode.')

    self.axis = axis
    self.dm_config = dm_config
    self.mode = mode
    self.epsilon = epsilon
    self.scale_initializer = tf.keras.initializers.get(scale_initializer)
    self.offset_initializer = tf.keras.initializers.get(offset_initializer)
    self.scale_regularizer = tf.keras.regularizers.get(scale_regularizer)
    self.offset_regularizer = tf.keras.regularizers.get(offset_regularizer)
    self.scale_constraint = tf.keras.constraints.get(scale_constraint)
    self.offset_constraint = tf.keras.constraints.get(offset_constraint)
    self.use_batch_normalization = use_batch_normalization
    self.trainable = trainable
    self.service_address = service_address

  @property
  def trainable(self):
    return self._trainable

  @trainable.setter
  def trainable(self, value):
    self._trainable = value

  @property
  def _param_dtype(self):
    # Raise parameters of fp16 batch norm to fp32
    if self.dtype == tf.dtypes.float16 or self.dtype == tf.dtypes.bfloat16:
      return tf.dtypes.float32
    else:
      return self.dtype or tf.dtypes.float32

  def _add_offset(self, name: typing.Text, shape):
    return self.add_weight(
        name=name,
        shape=shape,
        dtype=self._param_dtype,
        initializer=self.offset_initializer,
        regularizer=self.offset_regularizer,
        constraint=self.offset_constraint,
        trainable=True,
        experimental_autocast=False)

  def _add_scale(self, name: typing.Text, shape):
    return self.add_weight(
        name=name,
        shape=shape,
        dtype=self._param_dtype,
        initializer=self.scale_initializer,
        regularizer=self.scale_regularizer,
        constraint=self.scale_constraint,
        trainable=True,
        experimental_autocast=False)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if not input_shape.ndims:
      raise ValueError('Input has undefined rank:', input_shape)
    ndims = len(input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]

    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Validate axes
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('Duplicate axis: %s' % self.axis)

    axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
    for x in axis_to_dim:
      if axis_to_dim[x] is None:
        raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                         input_shape)
    self.input_spec = tf.keras.layers.InputSpec(ndim=ndims, axes=axis_to_dim)

    if len(axis_to_dim) == 1:
      # Single axis batch norm (most common/default use-case)
      param_shape = (list(axis_to_dim.values())[0],)
    else:
      # Parameter shape is the original shape but with 1 in all non-axis dims
      param_shape = [
          axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)
      ]

    self.mean_offset = self._add_offset('mean_offset', param_shape)
    self.mean_scale = self._add_scale('mean_scale', param_shape)

    if not self.use_batch_normalization:
      self.prior_offset = self._add_offset('prior_offset', param_shape)
      self.prior_scale = self._add_scale('prior_scale', param_shape)

    self.built = True

  def _get_training_value(self, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()
    return training

  def call(self, inputs, training=None):
    training = self._get_training_value(training)

    inputs_dtype = inputs.dtype.base_dtype
    if inputs_dtype in (tf.float16, tf.bfloat16):
      # Do all math in float32 if given 16-bit inputs for numeric stability.
      # In particular, it's very easy for variance to overflow in float16 and
      # for safety we also choose to cast bfloat16 to float32.
      inputs = tf.cast(inputs, tf.float32)

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]

    # Stops gradient update for the layers below in grow mode.
    # Intuitively when a new cluster is created, the gradients sent down to the
    # lower layers can disrupt the original weight, so it makes more sense to
    # freeze the other part when growing.
    inputs = tf.cond(
        tf.equal(self.mode, dm_ops.LOOKUP_WITH_GROW),
        lambda: tf.stop_gradient(inputs), lambda: inputs)

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

    def _broadcast(v):
      if (v is not None and len(v.shape) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return tf.reshape(v, broadcast_shape)
      return v

    mean_scale, mean_offset = _broadcast(self.mean_scale), _broadcast(
        self.mean_offset)
    if not self.use_batch_normalization:
      prior_scale, prior_offset = _broadcast(self.prior_scale), _broadcast(
          self.prior_offset)

    # Looks up mean and variances of from dynamic Gaussian memory.
    self.mean, self.variance, self.distance, self.cluster_id = (
        dm_ops.dynamic_gaussian_memory_lookup(
            inputs,
            self.mode,
            self.dm_config,
            'dm_lookup',
            service_address=self.service_address))

    if self.use_batch_normalization:
      outputs = tf.nn.batch_normalization(inputs, self.mean, self.variance,
                                          mean_offset, mean_scale, self.epsilon)
    else:
      outputs = dynamic_normalization(inputs, self.mean, self.variance,
                                      prior_offset, mean_offset, prior_scale,
                                      mean_scale, self.epsilon)
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    return outputs


def dynamic_normalization(x: tf.Tensor, mean: tf.Tensor, variance: tf.Tensor,
                          prior_offset: tf.Tensor, mean_offset: tf.Tensor,
                          prior_scale: tf.Tensor, mean_scale: tf.Tensor,
                          variance_epsilon: float):
  r"""Normalizes a tensor `x` based on the DyanmicNormalization formula.

  The normalization formula:
    ((|mean|^2 + p(mean)) / |x|^2 - 2 * g(mean)) / sqrt(variance)
  where
    p(mean) = prior_scale * mean + prior_offset
    g(mean) = mean_scale * mean + mean_offset.
  Here (prior_scale, prior_offset, mean_scale, mean_offset) are learnable
  parameters.

  Args:
    x: Input `Tensor` of arbitrary dimensionality.
    mean: A mean `Tensor`.
    variance: A variance `Tensor`.
    prior_offset: An offset `Tensor` for the prior term.
    mean_offset: An offset `Tensor` for estimating the mean value.
    prior_scale: A scale `Tensor` for the prior term.
    mean_scale: A scale `Tensor` for estimating the mean value.
    variance_epsilon: A small float number to avoid dividing by 0.

  Returns:
    The normalized, scaled, offset tensor.
  """
  with tf.name_scope('dynamic_normalization'):
    inv = tf.math.rsqrt(variance + variance_epsilon)

    # Computes (|mean|^2 + (scale * mean + offset)) / (|x|^2 + epsilon)
    dividend = tf.reduce_sum(tf.math.square(mean), -1, keepdims=True)
    dividend += prior_scale * mean + prior_offset
    divisor = tf.reduce_sum(
        tf.math.square(x), -1, keepdims=True) + variance_epsilon
    scale = dividend / divisor

    # Computes the dynamic normalization.
    return ((1 + scale) * x - 2 * (mean * mean_scale + mean_offset)) * inv
