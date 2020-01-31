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

"""A wrapper function to enable adversarial regularization to an Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structured_learning.configs as nsl_configs
import neural_structured_learning.lib as nsl_lib

import tensorflow as tf


def add_adversarial_regularization(estimator,
                                   optimizer_fn=None,
                                   adv_config=None):
  """Adds adversarial regularization to a `tf.estimator.Estimator`.

  The returned estimator will include the adversarial loss as a regularization
  term in its training objective, and will be trained using the optimizer
  provided by `optimizer_fn`. `optimizer_fn` (along with the hyperparameters)
  should be set to the same one used in the base `estimator`.

  If `optimizer_fn` is not set, a default optimizer `tf.train.AdagradOptimizer`
  with `learning_rate=0.05` will be used.

  Args:
    estimator: A `tf.estimator.Estimator` object, the base model.
    optimizer_fn: A function that accepts no arguments and returns an instance
      of `tf.train.Optimizer`. This optimizer (instead of the one used in
      `estimator`) will be used to train the model. If not specified, default to
      `tf.train.AdagradOptimizer` with `learning_rate=0.05`.
    adv_config: An instance of `nsl.configs.AdvRegConfig` that specifies various
      hyperparameters for adversarial regularization.

  Returns:
    A modified `tf.estimator.Estimator` object with adversarial regularization
    incorporated into its loss.
  """

  if not adv_config:
    adv_config = nsl_configs.AdvRegConfig()

  base_model_fn = estimator._model_fn  # pylint: disable=protected-access

  def adv_model_fn(features, labels, mode, params=None, config=None):
    """The adversarial-regularized model_fn.

    Args:
      features: This is the first item returned from the `input_fn` passed to
        `train`, `evaluate`, and `predict`. This should be a single `tf.Tensor`
        or `dict` of same.
      labels: This is the second item returned from the `input_fn` passed to
        `train`, `evaluate`, and `predict`. This should be a single `tf.Tensor`
        or dict of same (for multi-head models). If mode is
        `tf.estimator.ModeKeys.PREDICT`, `labels=None` will be passed. If the
        `model_fn`'s signature does not accept `mode`, the `model_fn` must still
        be able to handle `labels=None`.
      mode: Optional. Specifies if this is training, evaluation, or prediction.
        See `tf.estimator.ModeKeys`.
      params: Optional `dict` of hyperparameters. Will receive what is passed to
        Estimator in the `params` parameter. This allows users to configure
        Estimators from hyper parameter tuning.
      config: Optional `estimator.RunConfig` object. Will receive what is passed
        to Estimator as its `config` parameter, or a default value. Allows
        setting up things in the model_fn based on configuration such as
        `num_ps_replicas`, or `model_dir`. Unused currently.

    Returns:
      A `tf.estimator.EstimatorSpec` with adversarial regularization.
    """

    # Uses the same variable scope for calculating the original objective and
    # adversarial regularization.
    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),
                                     reuse=tf.compat.v1.AUTO_REUSE,
                                     auxiliary_name_scope=False):
      # If no 'params' is passed, then it is possible for base_model_fn not to
      # accept a 'params' argument. See documentation for tf.estimator.Estimator
      # for additional context.
      # pylint: disable=g-long-lambda
      spec_fn = ((lambda features: base_model_fn(
          features, labels, mode, params, config)) if params else (
              lambda features: base_model_fn(features, labels, mode, config)))

      original_spec = spec_fn(features)

      print("ORIGINAL_SPEC", original_spec)

      # Adversarial regularization only happens in training.
      if mode != tf.estimator.ModeKeys.TRAIN:
        return original_spec

      adv_neighbor, _ = nsl_lib.gen_adv_neighbor(features, original_spec.loss,
                                                 adv_config.adv_neighbor_config)

      # Runs the base model again to compute loss on adv_neighbor.
      adv_spec = spec_fn(adv_neighbor)

      final_loss = original_spec.loss + adv_config.multiplier * adv_spec.loss

      if not optimizer_fn:
        # Default to the Adagrad optimizer, the same as canned DNNEstimator.
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.05)
      else:
        optimizer = optimizer_fn()

      final_train_op = optimizer.minimize(
          loss=final_loss, global_step=tf.compat.v1.train.get_global_step())

    return original_spec._replace(loss=final_loss, train_op=final_train_op)

  # Replaces the model_fn while keeps other fields/methods in the estimator.
  estimator._model_fn = adv_model_fn  # pylint: disable=protected-access
  return estimator
