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
"""Distance functions used in Neural Structured Learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structured_learning.configs as configs

import tensorflow as tf


def _assert_multinomial_distribution(input_tensor, axis):
  """Assert input has valid multinomial distribution along `axis`."""
  sum_of_multinomial_distribution = tf.reduce_sum(
      input_tensor=input_tensor, axis=axis)
  return [
      tf.debugging.assert_non_negative(input_tensor),
      tf.debugging.assert_near(
          sum_of_multinomial_distribution,
          tf.constant(1.0),
          message='x and/or y is not a proper probability distribution'),
  ]


def _assert_valid_axis(ndims, axis):
  """Assert the condition `-ndims < axis <= ndims` if `axis` is not `None`."""
  if axis and (axis < -ndims or axis >= ndims):
    raise ValueError('axis = %d not in [%d, %d)' % (axis, -ndims, ndims))


def _kl_divergence_fn(true_dist, predicted_dist):
  epsilon = 1e-7  # A small increment to add to avoid taking a log of zero.
  return true_dist * tf.math.log(true_dist + epsilon) - true_dist * tf.math.log(
      predicted_dist + epsilon)


def kl_divergence(
    labels,
    predictions,
    axis=None,
    weights=1.0,
    scope=None,
    loss_collection=tf.compat.v1.GraphKeys.LOSSES,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a KL-divergence to the training procedure.

  For brevity, let `P = labels` and `Q = predictions`. The
  Kullback-Leibler divergence `KL(P||Q)` is:

  ```
  KL(P||Q) = P * log(P) - P * log(Q)
  ```

  Note: the function assumes that `predictions` and `labels` are the values of
  a multinomial distribution, i.e., each value is the probability of the
  corresponding class.

  For the usage of `weights` and `reduction`, please refer to `tf.losses`.

  Args:
    labels: `Tensor` of type `float32` or `float64`, with shape `[d1, ..., dN,
      num_classes]`, represents the target distribution.
    predictions: `Tensor` of the same type and shape as `labels`, represents
      the predicted distribution.
    axis: The dimension along which the KL divergence is computed. The values
      of `labels` and `predictions` along `axis` should meet the requirements
      of a multinomial distribution.
    weights: (optional) `Tensor` whose rank is either 0, or the same as that of
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: Collection to which the loss will be added.
    reduction: Type of reduction to apply to the loss.

  Returns:
    Weighted loss `float` `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`, otherwise, it is a scalar.
  Raises:
    InvalidArgumentError: If `labels` or `predictions` don't meet the
      requirements of a multinomial distribution.
    ValueError: If `axis` is `None`, if the shape of `predictions` doesn't
      match that of `labels`, or if the shape of `weights` is invalid.
  """
  with tf.compat.v1.name_scope(scope, 'kl_divergence',
                               (predictions, labels, weights)) as scope:
    labels = tf.cast(labels, tf.dtypes.float32)
    predictions = tf.cast(predictions, tf.dtypes.float32)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if axis is None:
      raise ValueError('You must specify "axis".')
    _assert_valid_axis(labels.get_shape().ndims, axis)
    assert_list = _assert_multinomial_distribution(
        labels, axis) + _assert_multinomial_distribution(predictions, axis)
    with tf.control_dependencies(assert_list):
      divergence_tensor = _kl_divergence_fn(labels, predictions)
      divergence = tf.reduce_sum(
          input_tensor=divergence_tensor, axis=(axis,), keepdims=True)
      return tf.compat.v1.losses.compute_weighted_loss(
          divergence, weights, scope, loss_collection, reduction=reduction)


def jensen_shannon_divergence(
    labels,
    predictions,
    axis=None,
    weights=1.0,
    scope=None,
    loss_collection=tf.compat.v1.GraphKeys.LOSSES,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a Jensen-Shannon divergence to the training procedure.

  For brevity, let `P = labels`, `Q = predictions`, `KL(P||Q)` be the
  Kullback-Leibler divergence as defined in the description of the
  `nsl.lib.kl_divergence` function.". The Jensen-Shannon divergence (JSD) is

  ```
  M = (P + Q) / 2
  JSD(P||Q) = KL(P||M) / 2 + KL(Q||M) / 2
  ```

  This function assumes that `predictions` and `labels` are the values of a
  multinomial distribution, i.e., each value is the probability of the
  corresponding class.

  For the usage of `weights` and `reduction`, please refer to `tf.losses`.

  Args:
    labels: `Tensor` of type `float32` or `float64`, with shape `[d1, ..., dN,
      num_classes]`, represents the target distribution.
    predictions: `Tensor` of the same type and shape as `labels`, represents
      the predicted distribution.
    axis: The dimension along which the Jensen-Shannon divergence is computed.
      The values of `labels` and `predictions` along `axis` should meet the
      requirements of a multinomial distribution.
    weights: (optional) `Tensor` whose rank is either 0, or the same as that of
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: Collection to which the loss will be added.
    reduction: Type of reduction to apply to the loss.

  Returns:
    Weighted loss `float` `Tensor`. If `reduction` is
    `tf.compat.v1.losses.Reduction.MEAN`, this has the same shape as `labels`,
    otherwise, it is a scalar.
  Raises:
    InvalidArgumentError: If `labels` or `predictions` don't meet the
      requirements of a multinomial distribution.
    ValueError: If `axis` is `None`, the shape of `predictions` doesn't match
      that of `labels`, or if the shape of `weights` is invalid.
  """
  with tf.compat.v1.name_scope(scope, 'jensen_shannon_divergence',
                               (predictions, labels, weights)) as scope:
    labels = tf.cast(labels, tf.dtypes.float32)
    predictions = tf.cast(predictions, tf.dtypes.float32)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if axis is None:
      raise ValueError('You must specify "axis".')
    _assert_valid_axis(labels.get_shape().ndims, axis)
    assert_list = _assert_multinomial_distribution(
        labels, axis) + _assert_multinomial_distribution(predictions, axis)
    with tf.control_dependencies(assert_list):
      means = 0.5 * (labels + predictions)
      divergence_tensor = 0.5 * _kl_divergence_fn(
          labels, means) + 0.5 * _kl_divergence_fn(predictions, means)
      divergence = tf.reduce_sum(
          input_tensor=divergence_tensor, axis=(axis,), keepdims=True)
      return tf.compat.v1.losses.compute_weighted_loss(
          divergence, weights, scope, loss_collection, reduction=reduction)


def _apply_transform(batched_tensor, transform_type, axis=None):
  """Applies the given transform function to `batched_tensor` along `axis`."""
  if transform_type == configs.TransformType.SOFTMAX:
    return tf.nn.softmax(batched_tensor, axis=axis)
  else:
    raise ValueError('Invalid TransformType %s.' % transform_type)


def _select_distance_fn(key):
  """Selects the distance function."""
  if key == configs.DistanceType.L1:
    return tf.compat.v1.losses.absolute_difference
  elif key == configs.DistanceType.L2:
    return tf.compat.v1.losses.mean_squared_error
  elif key == configs.DistanceType.COSINE:
    return tf.compat.v1.losses.cosine_distance
  elif key == configs.DistanceType.JENSEN_SHANNON_DIVERGENCE:
    return jensen_shannon_divergence
  elif key == configs.DistanceType.KL_DIVERGENCE:
    return kl_divergence
  else:
    raise ValueError('Invalid configs.DistanceType %s.' % key)


def _is_axis_required_in_distance_fn(key):
  return key in (configs.DistanceType.COSINE,
                 configs.DistanceType.JENSEN_SHANNON_DIVERGENCE,
                 configs.DistanceType.KL_DIVERGENCE)


def _is_reduced_by_average(key):
  return key in (tf.compat.v1.losses.Reduction.MEAN,
                 tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                 tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 tf.compat.v1.losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)


def pairwise_distance_wrapper(sources,
                              targets,
                              weights=1.0,
                              distance_config=None):
  """A wrapper to compute the pairwise distance between `sources` and `targets`.

  `distances = weights * distance_config.distance_type(sources, targets)`

  This wrapper calculates the weighted distance between `(sources, targets)`
  pairs, and provides an option to return the distance as the sum over the
  difference along the given axis, when vector based distance is needed.

  For the usage of `weights` and `reduction`, please refer to `tf.losses`. For
  the usage of `sum_over_axis`, see the following examples:

  Given target tensors with shape `[batch_size, features]`, the reduction set to
  `tf.compat.v1.losses.Reduction.MEAN`, and `sum_over_axis` set to the last
  dimension, the weighted average distance of sample pairs will be returned.
  For example: With a distance_config('L2', sum_over_axis=-1), the distance
  between [[1, 1], [2, 2], [0, 2], [5, 5]] and [[1, 1], [0, 2], [4, 4], [1, 4]]
  will be {(0+0) + (4+0) + (16+4) + (16+1)}/4 = 10.25

  If `sum_over_axis` is `None`, the weighted average distance of feature pairs
  (instead of sample pairs) will be returned. For example: With a
  distance_config('L2'), the distance between
  [[1, 1], [2, 2], [0, 2], [5, 5]] and [[1, 1], [0, 2], [4, 4], [1, 4]] will be
  {(0+0) + (4+0) + (16+4) + (16+1)}/8 = 5.125

  If `transform_fn` is not `None`, the transform function is applied to both
  `sources` and `targets` before computing the distance. For example:
  `distance_config('KL_DIVERGENCE', sum_over_axis=-1, transform_fn='SOFTMAX')`
  treats `sources` and `targets` as logits, and computes the KL-divergence
  between the two probability distributions.

  Args:
    sources: `Tensor` of type `float32` or `float64`.
    targets: `Tensor` of the same type and shape as `sources`.
    weights: (optional) `Tensor` whose rank is either 0, or the same as that of
      `targets`, and must be broadcastable to `targets` (i.e., all dimensions
      must be either `1`, or the same as the corresponding distance dimension).
    distance_config: An instance of `nsl.configs.DistanceConfig` that contains
      the following configuration (or hyperparameters) for computing distances:
      (a) `distance_type`: Type of distance function to apply.
      (b) `reduction`: Type of distance reduction. See `tf.losses.Reduction`.
      (c) `sum_over_axis`: (optional) The distance is the sum over the
        difference along the specified axis. Note that if `sum_over_axis` is not
        `None` and the rank of `weights` is non-zero, then the size of `weights`
        along `sum_over_axis` must be 1.
      (d) `transform_fn`: (optional) If set, both `sources` and `targets` will
        be transformed before calculating the distance. If set to 'SOFTMAX', it
        will be performed on the axis specified by 'sum_over_axis', or -1 if the
        axis is not specified. If `None`, the default distance config will be
        used.

  Returns:
    Weighted distance scalar `Tensor`. If `reduction` is
      `tf.compat.v1.losses.Reduction.MEAN`, this has the same shape as
      `targets`.
  Raises:
    ValueError: If the shape of targets doesn't match that of sources, or if the
      shape of weights is invalid.
    TypeError: If the distance function gets an unexpected keyword argument.
  """
  if distance_config is None:
    distance_config = configs.DistanceConfig()  # Default configs.

  tf.compat.v1.losses.Reduction.validate(distance_config.reduction)

  if distance_config.transform_fn is not configs.TransformType.NONE:
    sources = _apply_transform(sources, distance_config.transform_fn,
                               distance_config.sum_over_axis)
    targets = _apply_transform(targets, distance_config.transform_fn,
                               distance_config.sum_over_axis)

  sum_over_axis = distance_config.sum_over_axis
  # Validates the `sum_over_axis`
  _assert_valid_axis(sources.get_shape().ndims, sum_over_axis)
  distance_fn = _select_distance_fn(distance_config.distance_type)
  if distance_config.distance_type == configs.DistanceType.COSINE:
    # Cosine distance function assumes input tensors have been unit-normalized
    sources = tf.nn.l2_normalize(sources, axis=sum_over_axis)
    targets = tf.nn.l2_normalize(targets, axis=sum_over_axis)
  if _is_axis_required_in_distance_fn(distance_config.distance_type):
    distances = distance_fn(
        labels=sources,
        predictions=targets,
        weights=weights,
        axis=sum_over_axis,
        reduction=distance_config.reduction,
        loss_collection=None)
  else:
    distances = distance_fn(
        labels=sources,
        predictions=targets,
        weights=weights,
        reduction=distance_config.reduction,
        loss_collection=None)
    if sum_over_axis is not None and _is_reduced_by_average(
        distance_config.reduction):
      # The distance is divided by the size of targets tensor, so we need to
      # rescale the distance by multiplying the size of axis. Note, the distance
      # function with `axis` as a required argument (e.g., consine distance)
      # does not need to be rescaled.
      weights = tf.convert_to_tensor(value=weights)
      weights_shape = weights.get_shape().as_list()
      if weights_shape and weights_shape[sum_over_axis] != 1:
        raise ValueError('Shape of weights along the axis %d must be 1.' %
                         sum_over_axis)
      distances *= sources.shape.dims[sum_over_axis].value
  return distances
