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
"""A wrapper function to enable graph-based regularization for an Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import neural_structured_learning.configs as configs
from neural_structured_learning.lib import distances
from neural_structured_learning.lib import utils

import tensorflow as tf


def add_graph_regularization(estimator,
                             embedding_fn,
                             optimizer_fn=None,
                             graph_reg_config=None):
  """Adds graph regularization to a `tf.estimator.Estimator`.

  Args:
    estimator: An object of type `tf.estimator.Estimator`.
    embedding_fn: A function that accepts the input layer (dictionary of feature
      names and corresponding batched tensor values) as its first argument,
      an instance of `tf.estimator.ModeKeys` as its second argument to indicate
      if the mode is training, evaluation, or prediction, and an optional third
      argument named `params` which is a `dict` similar to the `params` argument
      of `tf.estimator.Estimator`'s `model_fn`, and returns the corresponding
      embeddings or logits to be used for graph regularization. The `params`
      argument will receive what was passed to `estimator` at the time of its
      creation as its `params` argument.
    optimizer_fn: A function that accepts no arguments and returns an instance
      of `tf.train.Optimizer`.
    graph_reg_config: An instance of `nsl.configs.GraphRegConfig` that specifies
      various hyperparameters for graph regularization.

  Returns:
    A modified `tf.estimator.Estimator` object with graph regularization
    incorporated into its loss.
  """

  if not graph_reg_config:
    graph_reg_config = configs.GraphRegConfig()

  base_model_fn = estimator._model_fn  # pylint: disable=protected-access
  try:
    base_model_fn_args = inspect.signature(base_model_fn).parameters.keys()
  except AttributeError:  # For Python 2 compatibility
    base_model_fn_args = inspect.getargspec(base_model_fn).args  # pylint: disable=deprecated-method

  def graph_reg_model_fn(features, labels, mode, params=None, config=None):
    """The graph-regularized model function.

    Args:
      features: This is the first item returned from the `input_fn` passed to
        `train`, `evaluate`, and `predict`. This should be a dictionary
        containing sample features as well as corresponding neighbor features
        and neighbor weights.
      labels: This is the second item returned from the `input_fn` passed to
        `train`, `evaluate`, and `predict`. This should be a single `Tensor` or
        `dict` of same (for multi-head models). If mode is
        `tf.estimator.ModeKeys.PREDICT`, `labels=None` will be passed. If the
        `model_fn`'s signature does not accept `mode`, the `model_fn` must still
        be able to handle `labels=None`.
      mode: Optional. Specifies if this is training, evaluation, or prediction.
        See `tf.estimator.ModeKeys`.
      params: Optional `dict` of hyperparameters. Will receive what is passed to
        Estimator in the `params` parameter. This allows users to configure
        Estimators from hyper parameter tuning.
      config: Optional `tf.estimator.RunConfig` object. Will receive what is
        passed to Estimator as its `config` parameter, or a default value.
        Allows setting up things in the `model_fn` based on configuration such
        as `num_ps_replicas`, or `model_dir`. Unused currently.

    Returns:
      A `tf.estimator.EstimatorSpec` with graph regularization.
    """
    # Parameters 'params' and 'config' are optional. If they are not passed,
    # then it is possible for base_model_fn not to accept these arguments.
    # See documentation for tf.estimator.Estimator for additional context.
    kwargs = {'mode': mode}
    embedding_fn_kwargs = dict()
    if 'params' in base_model_fn_args:
      kwargs['params'] = params
      embedding_fn_kwargs['params'] = params
    if 'config' in base_model_fn_args:
      kwargs['config'] = config

    # Uses the same variable scope for calculating the original objective and
    # the graph regularization loss term.
    with tf.compat.v1.variable_scope(
        tf.compat.v1.get_variable_scope(),
        reuse=tf.compat.v1.AUTO_REUSE,
        auxiliary_name_scope=False):
      nbr_features = dict()
      nbr_weights = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        # Extract sample features, neighbor features, and neighbor weights if we
        # are in training mode.
        sample_features, nbr_features, nbr_weights = (
            utils.unpack_neighbor_features(features,
                                           graph_reg_config.neighbor_config))
      else:
        # Otherwise, we strip out all neighbor features and use just the
        # sample's features.
        sample_features = utils.strip_neighbor_features(
            features, graph_reg_config.neighbor_config)

      base_spec = base_model_fn(sample_features, labels, **kwargs)

      has_nbr_inputs = nbr_weights is not None and nbr_features

      # Graph regularization happens only if all the following conditions are
      # satisfied:
      # - the mode is training
      # - neighbor inputs exist
      # - the graph regularization multiplier is greater than zero.
      # So, return early if any of these conditions is false.
      if (not has_nbr_inputs or mode != tf.estimator.ModeKeys.TRAIN or
          graph_reg_config.multiplier <= 0):
        return base_spec

      # Compute sample embeddings.
      sample_embeddings = embedding_fn(sample_features, mode,
                                       **embedding_fn_kwargs)

      # Compute the embeddings of the neighbors.
      nbr_embeddings = embedding_fn(nbr_features, mode, **embedding_fn_kwargs)

      replicated_sample_embeddings = utils.replicate_embeddings(
          sample_embeddings, graph_reg_config.neighbor_config.max_neighbors)

      # Compute the distance between the sample embeddings and each of their
      # corresponding neighbor embeddings.
      graph_loss = distances.pairwise_distance_wrapper(
          replicated_sample_embeddings,
          nbr_embeddings,
          weights=nbr_weights,
          distance_config=graph_reg_config.distance_config)
      scaled_graph_loss = graph_reg_config.multiplier * graph_loss
      tf.compat.v1.summary.scalar('loss/scaled_graph_loss', scaled_graph_loss)

      supervised_loss = base_spec.loss
      tf.compat.v1.summary.scalar('loss/supervised_loss', supervised_loss)

      total_loss = supervised_loss + scaled_graph_loss

      if not optimizer_fn:
        # Default to Adagrad optimizer, the same as the canned DNNEstimator.
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.05)
      else:
        optimizer = optimizer_fn()
      train_op = optimizer.minimize(
          loss=total_loss, global_step=tf.compat.v1.train.get_global_step())
      update_ops = tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.UPDATE_OPS)
      if update_ops:
        train_op = tf.group(train_op, *update_ops)

    return base_spec._replace(loss=total_loss, train_op=train_op)

  # Replaces the model_fn while keeping other fields/methods in the estimator.
  estimator._model_fn = graph_reg_model_fn  # pylint: disable=protected-access
  return estimator
