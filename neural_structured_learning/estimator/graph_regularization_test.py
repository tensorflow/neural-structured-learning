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
"""Tests for nsl.estimator.graph_regularization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import neural_structured_learning.configs as nsl_configs
import neural_structured_learning.estimator as nsl_estimator

import numpy as np
import tensorflow as tf
from tensorflow import estimator as tf_estimator

from google.protobuf import text_format
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


FEATURE_NAME = 'x'
LABEL_NAME = 'y'
NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'
WEIGHT_VARIABLE = 'linear/linear_model/' + FEATURE_NAME + '/weights'
BIAS_VARIABLE = 'linear/linear_model/bias_weights'
LEARNING_RATE = 0.01


def make_feature_spec(input_shape, max_neighbors):
  """Returns a feature spec that can be used to parse tf.train.Examples.

  Args:
    input_shape: A list of integers representing the shape of the input feature
      and corresponding neighbor features.
    max_neighbors: The maximum neighbors per sample to be used for graph
      regularization.
  """
  feature_spec = {
      FEATURE_NAME:
          tf.FixedLenFeature(input_shape, tf.float32),
      LABEL_NAME:
          tf.FixedLenFeature([1], tf.float32, default_value=tf.constant([0.0])),
  }
  for i in range(max_neighbors):
    nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, FEATURE_NAME)
    nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i, NBR_WEIGHT_SUFFIX)
    feature_spec[nbr_feature_key] = tf.FixedLenFeature(input_shape, tf.float32)
    feature_spec[nbr_weight_key] = tf.FixedLenFeature([1],
                                                      tf.float32,
                                                      default_value=tf.constant(
                                                          [0.0]))
  return feature_spec


def single_example_input_fn(example_proto, input_shape, max_neighbors):
  """Returns an input_fn."""

  def make_parse_example_fn(feature_spec):
    """Creates a function parse_example function.

    Args:
      feature_spec: A dictionary of features for parsing an input
        tf.train.Example.

    Returns:
      A parse_example function.
    """

    def parse_example(serialized_example_proto):
      """Extracts relevant fields from the example_proto."""
      return tf.parse_single_example(serialized_example_proto, feature_spec)

    return parse_example

  def input_fn():
    # Construct a tf.data.Dataset from the given Example.
    example = text_format.Parse(example_proto, tf.train.Example())
    serialized_example = example.SerializeToString()
    dataset = tf.data.Dataset.from_tensors(
        tf.convert_to_tensor(serialized_example))
    dataset = dataset.map(
        make_parse_example_fn(make_feature_spec(input_shape, max_neighbors)))
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    batch_features = iterator.get_next()
    return batch_features, batch_features.pop(LABEL_NAME)

  return input_fn


class GraphRegularizationTest(tf.test.TestCase):

  def setUp(self):
    super(GraphRegularizationTest, self).setUp()
    self.model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self.model_dir:
      shutil.rmtree(self.model_dir)
    super(GraphRegularizationTest, self).tearDown()

  def build_linear_regressor(self, weight, weight_shape, bias, bias_shape):
    with tf.Graph().as_default():
      # Use a partitioner that is known a priori because canned Estimators
      # default to using one otherwise. This allows tests to access variables
      # used in the underlying Estimator.
      tf.get_variable(
          name=WEIGHT_VARIABLE,
          shape=weight_shape,
          initializer=weight,
          partitioner=tf.fixed_size_partitioner(1))
      tf.get_variable(
          name=BIAS_VARIABLE,
          shape=bias_shape,
          initializer=bias,
          partitioner=tf.fixed_size_partitioner(1))
      tf.Variable(100, name=tf.GraphKeys.GLOBAL_STEP, dtype=tf.int64)

      with tf.Session() as sess:
        sess.run([tf.global_variables_initializer()])
        tf.train.Saver().save(sess, os.path.join(self.model_dir, 'model.ckpt'))

    fc = tf.feature_column.numeric_column(
        FEATURE_NAME, shape=np.array(weight).shape)
    return tf_estimator.LinearRegressor(
        feature_columns=(fc,), model_dir=self.model_dir, optimizer='SGD')

  @test_util.run_v1_only('Requires tf.get_variable')
  def test_graph_reg_wrapper_no_training(self):
    """Test that predictions are unaffected when there is no training."""
    # Base model: y = x + 2
    base_est = self.build_linear_regressor(
        weight=[[1.0]], weight_shape=[1, 1], bias=[2.0], bias_shape=[1])

    def embedding_fn(features, unused_mode):
      # Apply the same model, i.e, y = x + 2.
      # Use broadcasting to do element-wise addition.
      return tf.math.add(features[FEATURE_NAME], [2.0])

    graph_reg_config = nsl_configs.make_graph_reg_config(max_neighbors=1)
    graph_reg_est = nsl_estimator.add_graph_regularization(
        base_est, embedding_fn, graph_reg_config=graph_reg_config)

    # Consider only one neighbor for the input sample.
    example = """
                features {
                  feature {
                    key: "x"
                    value: { float_list { value: [ 1.0 ] } }
                  }
                  feature {
                    key: "NL_nbr_0_x"
                    value: { float_list { value: [ 2.0 ] } }
                  }
                  feature {
                    key: "NL_nbr_0_weight"
                    value: { float_list { value: 1.0 } }
                  }
               }
              """

    input_fn = single_example_input_fn(
        example, input_shape=[1], max_neighbors=0)
    predictions = graph_reg_est.predict(input_fn=input_fn)
    predicted_scores = [x['predictions'] for x in predictions]
    self.assertAllClose([[3.0]], predicted_scores)

  def _train_and_check_params(self, example, max_neighbors, weight, bias,
                              expected_grad_from_weight,
                              expected_grad_from_bias):
    """Runs training for one step and verifies gradient-based updates."""

    def embedding_fn(features, unused_mode):
      # Computes y = w*x
      with tf.variable_scope(
          tf.get_variable_scope(),
          reuse=tf.AUTO_REUSE,
          auxiliary_name_scope=False):
        weight_tensor = tf.reshape(
            tf.get_variable(
                WEIGHT_VARIABLE,
                shape=[2, 1],
                partitioner=tf.fixed_size_partitioner(1)),
            shape=[-1, 2])

      x_tensor = tf.reshape(features[FEATURE_NAME], shape=[-1, 2])
      return tf.reduce_sum(
          tf.multiply(weight_tensor, x_tensor), 1, keep_dims=True)

    def optimizer_fn():
      return tf.train.GradientDescentOptimizer(LEARNING_RATE)

    base_est = self.build_linear_regressor(
        weight=weight, weight_shape=[2, 1], bias=bias, bias_shape=[1])

    graph_reg_config = nsl_configs.make_graph_reg_config(
        max_neighbors=max_neighbors, multiplier=1)
    graph_reg_est = nsl_estimator.add_graph_regularization(
        base_est, embedding_fn, optimizer_fn, graph_reg_config=graph_reg_config)

    input_fn = single_example_input_fn(
        example, input_shape=[2], max_neighbors=max_neighbors)
    graph_reg_est.train(input_fn=input_fn, steps=1)

    # Compute the new bias and weight values based on the gradients.
    expected_bias = bias - LEARNING_RATE * (expected_grad_from_bias)
    expected_weight = weight - LEARNING_RATE * (expected_grad_from_weight)

    # Check that the parameters of the linear regressor have the correct values.
    self.assertAllClose(expected_bias,
                        graph_reg_est.get_variable_value(BIAS_VARIABLE))
    self.assertAllClose(expected_weight,
                        graph_reg_est.get_variable_value(WEIGHT_VARIABLE))

  @test_util.run_v1_only('Requires tf.get_variable')
  def test_graph_reg_wrapper_one_neighbor_with_training(self):
    """Tests that the loss during training includes graph regularization."""
    # Base model: y = w*x+b = 4*x1 + 3*x2 + 2
    weight = np.array([[4.0], [3.0]], dtype=np.float32)
    bias = np.array([2.0], dtype=np.float32)
    # Expected y value.
    x0, y0 = np.array([[1.0, 1.0]]), np.array([8.0])
    neighbor0 = np.array([[0.5, 1.5]])

    example = """
                features {
                  feature {
                    key: "x"
                    value: { float_list { value: [ 1.0, 1.0 ] } }
                  }
                  feature {
                    key: "NL_nbr_0_x"
                    value: { float_list { value: [ 0.5, 1.5 ] } }
                  }
                  feature {
                    key: "NL_nbr_0_weight"
                    value: { float_list { value: 1.0 } }
                  }
                  feature {
                    key: "y"
                    value: { float_list { value: 8.0 } }
                  }
                }
              """

    # Compute the gradients on the original example using the updated loss term,
    # which includes the supervised loss as well as the graph loss.
    orig_pred = np.dot(x0, weight) + bias  # [9.0]

    # Based on the implementation of embedding_fn inside
    # _train_and_check_params.
    x0_embedding = np.dot(x0, weight)
    neighbor0_embedding = np.dot(neighbor0, weight)

    # The graph loss term is (x0_embedding - neighbor0_embedding)^2
    orig_grad_w = 2 * (orig_pred - y0) * x0.T + 2 * (
        x0_embedding - neighbor0_embedding) * (x0 -
                                               neighbor0).T  # [[2.5], [1.5]]
    orig_grad_b = 2 * (orig_pred - y0).reshape((1,))  # [2.0]

    self._train_and_check_params(example, 1, weight, bias, orig_grad_w,
                                 orig_grad_b)

  @test_util.run_v1_only('Requires tf.get_variable')
  def test_graph_reg_wrapper_two_neighbors_with_training(self):
    """Tests that the loss during training includes graph regularization."""
    # Base model: y = w*x+b = 4*x1 + 3*x2 + 2
    weight = np.array([[4.0], [3.0]], dtype=np.float32)
    bias = np.array([2.0], dtype=np.float32)
    # Expected y value.
    x0, y0 = np.array([[1.0, 1.0]]), np.array([8.0])
    neighbor0 = np.array([[0.5, 1.5]])
    neighbor1 = np.array([[0.75, 1.25]])

    example = """
                features {
                  feature {
                    key: "x"
                    value: { float_list { value: [ 1.0, 1.0 ] } }
                  }
                  feature {
                    key: "NL_nbr_0_x"
                    value: { float_list { value: [ 0.5, 1.5 ] } }
                  }
                  feature {
                    key: "NL_nbr_0_weight"
                    value: { float_list { value: 1.0 } }
                  }
                  feature {
                    key: "NL_nbr_1_x"
                    value: { float_list { value: [ 0.75, 1.25 ] } }
                  }
                  feature {
                    key: "NL_nbr_1_weight"
                    value: { float_list { value: 1.0 } }
                  }
                  feature {
                    key: "y"
                    value: { float_list { value: 8.0 } }
                  }
                }
              """

    # Compute the gradients on the original example using the updated loss term,
    # which includes the supervised loss as well as the graph loss.
    orig_pred = np.dot(x0, weight) + bias  # [9.0]

    # Based on the implementation of embedding_fn inside
    # _train_and_check_params.
    x0_embedding = np.dot(x0, weight)
    neighbor0_embedding = np.dot(neighbor0, weight)
    neighbor1_embedding = np.dot(neighbor1, weight)

    # Vertify that the loss includes the supervised loss as well as the graph
    # loss by computing the gradients of the loss.

    grad_w_supervised_loss = 2 * (orig_pred - y0) * x0.T  # [[2.0], [2.0]]

    # The distance metric for the graph loss is 'L2'. So, the graph loss term is
    # [(x0_embedding - neighbor0_embedding)^2 +
    #  (x0_embedding - neighbor1_embedding)^2] / 2
    grad_w_graph_loss = ((x0_embedding - neighbor0_embedding) *
                         (x0 - neighbor0).T) + (
                             (x0_embedding - neighbor1_embedding) *
                             (x0 - neighbor1).T)  # [[0.3125], [-0.3125]]
    orig_grad_w = grad_w_supervised_loss + grad_w_graph_loss
    orig_grad_b = 2 * (orig_pred - y0).reshape((1,))  # [2.0]

    self._train_and_check_params(example, 2, weight, bias, orig_grad_w,
                                 orig_grad_b)

  def _train_and_check_eval_results(self, train_example, test_example,
                                    max_neighbors, weight, bias):
    """Verifies evaluation results for the graph-regularized model."""

    def embedding_fn(features, unused_mode):
      # Computes y = w*x
      with tf.variable_scope(
          tf.get_variable_scope(),
          reuse=tf.AUTO_REUSE,
          auxiliary_name_scope=False):
        weight_tensor = tf.reshape(
            tf.get_variable(
                WEIGHT_VARIABLE,
                shape=[2, 1],
                partitioner=tf.fixed_size_partitioner(1)),
            shape=[-1, 2])

      x_tensor = tf.reshape(features[FEATURE_NAME], shape=[-1, 2])
      return tf.reduce_sum(
          tf.multiply(weight_tensor, x_tensor), 1, keep_dims=True)

    def optimizer_fn():
      return tf.train.GradientDescentOptimizer(LEARNING_RATE)

    base_est = self.build_linear_regressor(
        weight=weight, weight_shape=[2, 1], bias=bias, bias_shape=[1])

    graph_reg_config = nsl_configs.make_graph_reg_config(
        max_neighbors=max_neighbors, multiplier=1)
    graph_reg_est = nsl_estimator.add_graph_regularization(
        base_est, embedding_fn, optimizer_fn, graph_reg_config=graph_reg_config)

    train_input_fn = single_example_input_fn(
        train_example, input_shape=[2], max_neighbors=max_neighbors)
    graph_reg_est.train(input_fn=train_input_fn, steps=1)

    # Evaluating the graph-regularized model should yield the same results
    # as evaluating the base model because model paramters are shared.
    eval_input_fn = single_example_input_fn(
        test_example, input_shape=[2], max_neighbors=0)
    graph_eval_results = graph_reg_est.evaluate(input_fn=eval_input_fn)
    base_eval_results = base_est.evaluate(input_fn=eval_input_fn)
    self.assertAllClose(base_eval_results, graph_eval_results)

  @test_util.run_v1_only('Requires tf.get_variable')
  def test_graph_reg_model_evaluate(self):
    weight = np.array([[4.0], [-3.0]])
    bias = np.array([0.0], dtype=np.float32)

    train_example = """
                features {
                  feature {
                    key: "x"
                    value: { float_list { value: [ 2.0, 3.0 ] } }
                  }
                  feature {
                    key: "NL_nbr_0_x"
                    value: { float_list { value: [ 2.5, 3.0 ] } }
                  }
                  feature {
                    key: "NL_nbr_0_weight"
                    value: { float_list { value: 1.0 } }
                  }
                  feature {
                    key: "NL_nbr_1_x"
                    value: { float_list { value: [ 2.0, 2.0 ] } }
                  }
                  feature {
                    key: "NL_nbr_1_weight"
                    value: { float_list { value: 1.0 } }
                  }
                  feature {
                    key: "y"
                    value: { float_list { value: 0.0 } }
                  }
                }
              """

    test_example = """
                features {
                  feature {
                    key: "x"
                    value: { float_list { value: [ 4.0, 2.0 ] } }
                  }
                  feature {
                    key: "y"
                    value: { float_list { value: 4.0 } }
                  }
                }
              """
    self._train_and_check_eval_results(
        train_example, test_example, max_neighbors=2, weight=weight, bias=bias)

  @test_util.run_v1_only('Requires tf.GraphKeys')
  def test_graph_reg_wrapper_saving_batch_statistics(self):
    """Verifies that batch statistics in batch-norm layers are saved."""

    def optimizer_fn():
      return tf.train.GradientDescentOptimizer(0.005)

    def embedding_fn(features, mode, params=None):
      del params
      input_layer = features[FEATURE_NAME]
      with tf.compat.v1.variable_scope('hidden_layer', reuse=tf.AUTO_REUSE):
        hidden_layer = tf.compat.v1.layers.dense(
            input_layer, units=4, activation=lambda x: tf.abs(x) + 0.1)
        # The always-positive activation funciton is to make sure the batch mean
        # is non-zero.
        batch_norm_layer = tf.compat.v1.layers.batch_normalization(
            hidden_layer, training=(mode == tf_estimator.ModeKeys.TRAIN))
      return batch_norm_layer

    def model_fn(features, labels, mode, params=None, config=None):
      del params, config
      embeddings = embedding_fn(features, mode)
      with tf.compat.v1.variable_scope('logit', reuse=tf.AUTO_REUSE):
        logits = tf.compat.v1.layers.dense(embeddings, units=1)
      predictions = tf.argmax(logits, 1)
      if mode == tf_estimator.ModeKeys.PREDICT:
        return tf_estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'logits': logits,
                'predictions': predictions
            })

      loss = tf.losses.sigmoid_cross_entropy(labels, logits)
      if mode == tf_estimator.ModeKeys.EVAL:
        return tf_estimator.EstimatorSpec(mode=mode, loss=loss)

      optimizer = optimizer_fn()
      train_op = optimizer.minimize(
          loss, global_step=tf.compat.v1.train.get_global_step())
      update_ops = tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.UPDATE_OPS)
      train_op = tf.group(train_op, *update_ops)
      return tf_estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def input_fn():
      nbr_feature = '{}{}_{}'.format(NBR_FEATURE_PREFIX, 0, FEATURE_NAME)
      nbr_weight = '{}{}{}'.format(NBR_FEATURE_PREFIX, 0, NBR_WEIGHT_SUFFIX)
      features = {
          FEATURE_NAME: tf.constant([[0.1, 0.9], [-0.8, -0.2], [0.3, -0.7]]),
          nbr_feature: tf.constant([[0.1, 0.89], [-0.81, -0.2], [0.3, -0.69]]),
          nbr_weight: tf.constant([[0.9], [0.8], [0.7]]),
      }
      labels = tf.constant([[1], [0], [1]])
      return tf.data.Dataset.from_tensor_slices((features, labels)).batch(3)

    base_est = tf_estimator.Estimator(
        model_fn, model_dir=self.model_dir, params=None)
    graph_reg_config = nsl_configs.make_graph_reg_config(
        max_neighbors=1, multiplier=1)
    graph_reg_est = nsl_estimator.add_graph_regularization(
        base_est, embedding_fn, optimizer_fn, graph_reg_config=graph_reg_config)
    graph_reg_est.train(input_fn, steps=1)

    moving_mean = graph_reg_est.get_variable_value(
        'hidden_layer/batch_normalization/moving_mean')
    moving_variance = graph_reg_est.get_variable_value(
        'hidden_layer/batch_normalization/moving_variance')
    self.assertNotAllClose(moving_mean, np.zeros(moving_mean.shape))
    self.assertNotAllClose(moving_variance, np.ones(moving_variance.shape))


if __name__ == '__main__':
  tf.test.main()
