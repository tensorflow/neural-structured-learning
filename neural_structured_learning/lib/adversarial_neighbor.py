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

"""Generates adversarial neighbors.

This file provides the class(es) and the corresponding functional interface(s)
for generating adversarial neighbors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from neural_structured_learning.lib import abstract_gen_neighbor as abs_gen
from neural_structured_learning.lib import utils

import tensorflow as tf


class _GenAdvNeighbor(abs_gen.GenNeighbor):
  """Class for generating adversarial neighbors.

  The core of this class implements the operation:
  `adv_neighbor = input_features + adv_step_size * final_grad`
  where `adv_step_size` is the step size (analogous to learning rate) for
  searching/calculating adversarial neighbor.

  Attributes:
    labeled_loss: a scalar (`tf.float32`) tensor calculated from true labels (or
      supervisions).
    adv_config: a `nsl.configs.AdvNeighborConfig` object.
    raise_invalid_gradient: (optional) a Boolean flag indicating whether to
      raise an error when gradients cannot be computed on some input features.
      There are three cases where gradients cannot be computed:  (1) The feature
        is a `tf.SparseTensor`. (2) The feature is in a non-differentiable
        `tf.DType`, like string or integer. (3) The feature is not involved in
        loss computation.  If set to False, those input without gradient will be
        ignored silently and not perturbed. (default=False)
  """

  def __init__(self,
               labeled_loss,
               adv_config,
               raise_invalid_gradient=False,
               gradient_tape=None):
    self._labeled_loss = labeled_loss
    self._adv_config = adv_config
    self._raise_invalid_gradient = raise_invalid_gradient
    self._gradient_tape = gradient_tape

  def _normalize_gradient(self, keyed_grads, feature_masks):
    """Masks the gradients and normalizes to the size specified in adv_config.

    Arguments:
      keyed_grads: A dictionary of (feature_name, Tensor) representing gradients
        on each feature.
      feature_masks: A dictionary of (feature_name, Tensor-compatible value)
        representing masks on each feature. A feature is not masked if its name
        is missing in this dictionary.

    Returns:
      perturbation: A dictionary of (feature_name, Tensor) representing the
        adversarial perturbation on that feature.

    Raises:
      ValueError: if 'raise_invalid_gradient' is set and gradients cannot be
        computed on some input features.
    """
    grads_to_concat = []
    dim_index_and_sizes = {}
    total_dims = 0
    for (key, grad) in keyed_grads.items():
      if grad is None:
        # Two cases that grad can be None:
        # (1) The feature is not differentiable, like strings, integer indices.
        # (2) The feature is not involved in loss computation.
        # In either case, no gradient will be calculated for this feature.
        if self._raise_invalid_gradient:
          raise ValueError('Cannot perturb feature ' + key)
        tf.compat.v1.logging.warn('Cannot perturb feature ' + key)
        continue

      # Guards against numerical errors. If the gradient is malformed (inf,
      # -inf, or NaN) on a dimension, replace it with 0, which has the effect of
      # not perturbing the original sample along that perticular dimension.
      grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
      # Applies feature masks if available.
      if key in feature_masks:
        grad *= tf.cast(feature_masks[key], grad.dtype)

      # The gradients are reshaped to 2-D (batch_size x total_feature_len;
      # sequence data will be processed in the same way) so they can be
      # concatenated and normalized across features. They will be reshaped back
      # to the original shape after normalization.
      feature_dim = tf.reduce_prod(input_tensor=grad.get_shape()[1:])
      grad = tf.reshape(grad, [-1, feature_dim])
      grads_to_concat.append(grad)
      dim_index_and_sizes[key] = (total_dims, total_dims + feature_dim)
      total_dims += feature_dim

    if not grads_to_concat:
      return {}  # no perturbation

    # Concatenates all the gradients so they can be normalized together.
    concat_grads = tf.concat(grads_to_concat, axis=-1)
    adv_perturbation = utils.maximize_within_unit_norm(
        concat_grads, self._adv_config.adv_grad_norm)
    adv_perturbation = self._adv_config.adv_step_size * adv_perturbation

    perturbation = {}
    for (key, grad) in keyed_grads.items():
      if key not in dim_index_and_sizes:
        continue
      dim_idx_begin, dim_idx_end = dim_index_and_sizes[key]
      sub_grad = adv_perturbation[:, dim_idx_begin:dim_idx_end]
      if grad.get_shape().rank > 2:
        sub_grad = tf.reshape(sub_grad, [-1] + grad.get_shape().as_list()[1:])
      perturbation[key] = sub_grad
    return perturbation

  def gen_neighbor(self, input_features):
    """Generates adversarial neighbors and the corresponding weights.

    This function perturbs only *dense* tensors to generate adversarial
    neighbors. No pertubation will be applied on sparse tensors  (e.g., string
    or int). Therefore, in the generated adversarial neighbors, the values of
    these sparse tensors will be kept the same as the input_features. In other
    words, if input_features is a dictionary mapping feature names to tensors,
    the dense features will be perturbed and the values of sparse features will
    remain the same.

    Arguments:
      input_features: a dense (float32) tensor or a dictionary of feature names
        and dense tensors. The shape of the tensor(s) should be either:
        (a) pointwise samples: [batch_size, feat_len], or
        (b) sequence samples: [batch_size, seq_len, feat_len]

    Returns:
      adv_neighbor: the perturbed example, with the same shape and structure as
        input_features
      adv_weight: a dense (float32) tensor with shape of [batch_size, 1],
        representing the weight for each neighbor

    Raises:
      ValueError: if some of the `input_features` cannot be perturbed due to
        (a) it is a `tf.SparseTensor`,
        (b) it has a non-differentiable type like string or integer, or
        (c) it is not involved in loss computation.
        This error is suppressed if `raise_invalid_gradient` is set to False
        (which is the default).
    """
    single_feature = not isinstance(input_features, dict)

    # Converts single-feature input to a dictionary to reuse following code.
    if single_feature:
      input_features = {'': input_features}

    sparse_features, dense_features = {}, {}
    for (key, feature) in input_features.items():
      if isinstance(feature, tf.Tensor):
        dense_features[key] = feature
      else:
        sparse_features[key] = feature

    if sparse_features:
      sparse_keys = str(sparse_features.keys())
      if self._raise_invalid_gradient:
        raise ValueError('Cannot perturb non-Tensor input: ' + sparse_keys)
      tf.compat.v1.logging.warn('Cannot perturb non-Tensor input: ' +
                                sparse_keys)

    if self._adv_config.feature_mask is None:
      feature_masks = {}  # missing key => no mask
    elif single_feature:
      feature_masks = {'': self._adv_config.feature_mask}
    else:
      feature_masks = self._adv_config.feature_mask

    # Computes the gradient of the loss w.r.t. each dense feature. The returned
    # value is a list of tensors, each with the same shape as the corresponding
    # feature tensor. The order of elements returned by .values() and .keys()
    # are guaranteed corresponding to each other.
    if self._gradient_tape is None:  # Assuming in graph mode, no tape required.
      grads = tf.gradients(
          ys=[self._labeled_loss], xs=list(dense_features.values()))
    else:
      grads = self._gradient_tape.gradient(self._labeled_loss,
                                           list(dense_features.values()))
    keyed_grads = dict(zip(dense_features.keys(), grads))
    perturbation = self._normalize_gradient(keyed_grads, feature_masks)

    # Sparse features are copied directly without perturbation.
    adv_neighbor = dict(sparse_features)
    for (key, feature) in dense_features.items():
      adv_neighbor[key] = tf.stop_gradient(
          feature if key not in perturbation else feature + perturbation[key])

    # Converts the perturbed examples back to their original format.
    if single_feature:
      adv_neighbor = adv_neighbor['']

    batch_size = tf.shape(
        input=adv_neighbor if single_feature else list(adv_neighbor.values())[0]
    )[0]
    adv_weight = tf.ones([batch_size, 1])

    return adv_neighbor, adv_weight


def gen_adv_neighbor(input_features,
                     labeled_loss,
                     config,
                     raise_invalid_gradient=False,
                     gradient_tape=None):
  """Functional interface of _GenAdvNeighbor.

  This function provides a tensor/config-in & tensor-out functional interface
  that does the following:
  (a) Instantiates '_GenAdvNeighbor'
  (b) Invokes 'gen_neighbor' method
  (c) Returns the adversarial neighbors generated.

  Arguments:
    input_features: a dense (float32) tensor or a dictionary of feature names
      and dense tensors. The shape of the tensor(s) should be either:
      (a) pointwise samples: [batch_size, feat_len], or
      (b) sequence samples: [batch_size, seq_len, feat_len]. Note that if the
        `input_features` is a dictionary, only dense (`float`) tensors in it
        will be perturbed and all other features (int, string, or sparse
        tensors) will be kept as-is in the returning `adv_neighbor`.
    labeled_loss: a scalar (float32) tensor calculated from true labels (or
      supervisions).
    config: `AdvNeighborConfig` object containing the following hyperparameters
      for generating adversarial samples. - 'feature_mask': mask (w/ 0-1 values)
      applied on graident. - 'adv_step_size': step size to find the adversarial
      sample. - 'adv_grad_norm': type of tensor norm to normalize the gradient.
    raise_invalid_gradient: (optional, default=False) a Boolean flag indicating
      whether to raise an error when gradients cannot be computed on some input
      features. There are three cases where gradients cannot be computed:  (1)
        The feature is a SparseTensor. (2) The feature has a non-differentiable
        `tf.DType`, like string or integer. (3) The feature is not involved in
        loss computation.  If set to False (default), those inputs without
        gradient will be ignored silently and not perturbed.
    gradient_tape: a `tf.GradientTape` object watching the calculation from
      `input_features` to `labeled_loss`. Can be omitted if running in graph
      mode.

  Returns:
    adv_neighbor: the perturbed example, with the same shape and structure as
      input_features
    adv_weight: a dense (float32) tensor with shape of [batch_size, 1],
      representing the weight for each neighbor

  Raises:
    ValueError: if `raise_invalid_gradient` is set and some of the input
      features cannot be perturbed. See `raise_invalid_gradient` for situations
      where this can happen.
  """
  adv_helper = _GenAdvNeighbor(labeled_loss, config, raise_invalid_gradient,
                               gradient_tape)
  return adv_helper.gen_neighbor(input_features)
