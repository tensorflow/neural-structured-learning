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
"""Tests for nsl.estimator.adversarial_regularization."""

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

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


FEATURE_NAME = 'x'
WEIGHT_VARIABLE = 'linear/linear_model/' + FEATURE_NAME + '/weights'
BIAS_VARIABLE = 'linear/linear_model/bias_weights'


def single_batch_input_fn(features, labels=None):
  def input_fn():
    inputs = features if labels is None else (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    return dataset.batch(len(features))
  return input_fn


class AdversarialRegularizationTest(tf.test.TestCase):

  def setUp(self):
    super(AdversarialRegularizationTest, self).setUp()
    self.model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self.model_dir:
      shutil.rmtree(self.model_dir)
    super(AdversarialRegularizationTest, self).tearDown()

  def build_linear_regressor(self, weight, bias):
    with tf.Graph().as_default():
      tf.Variable(weight, name=WEIGHT_VARIABLE)
      tf.Variable(bias, name=BIAS_VARIABLE)
      tf.Variable(100, name=tf.GraphKeys.GLOBAL_STEP, dtype=tf.int64)

      with tf.Session() as sess:
        sess.run([tf.global_variables_initializer()])
        tf.train.Saver().save(sess, os.path.join(self.model_dir, 'model.ckpt'))

    fc = tf.feature_column.numeric_column(FEATURE_NAME,
                                          shape=np.array(weight).shape)
    return tf.estimator.LinearRegressor(
        feature_columns=(fc,), model_dir=self.model_dir, optimizer='SGD')

  @test_util.run_v1_only('Requires tf.GraphKeys')
  def test_adversarial_wrapper_not_affecting_predictions(self):
    # base model: y = x + 2
    base_est = self.build_linear_regressor(weight=[[1.0]], bias=[2.0])
    adv_est = nsl_estimator.add_adversarial_regularization(base_est)
    input_fn = single_batch_input_fn({FEATURE_NAME: np.array([[1.0], [2.0]])})
    predictions = adv_est.predict(input_fn=input_fn)
    predicted_scores = [x['predictions'] for x in predictions]
    self.assertAllClose([[3.0], [4.0]], predicted_scores)

  @test_util.run_v1_only('Requires tf.GraphKeys')
  def test_adversarial_wrapper_adds_regularization(self):
    # base model: y = w*x+b = 4*x1 + 3*x2 + 2
    weight = np.array([[4.0], [3.0]], dtype=np.float32)
    bias = np.array([2.0], dtype=np.float32)
    x0, y0 = np.array([[1.0, 1.0]]), np.array([8.0])
    adv_step_size = 0.1
    learning_rate = 0.01

    base_est = self.build_linear_regressor(weight=weight, bias=bias)
    adv_config = nsl_configs.make_adv_reg_config(
        multiplier=1.0,  # equal weight on original and adv examples
        adv_step_size=adv_step_size)
    adv_est = nsl_estimator.add_adversarial_regularization(
        base_est,
        optimizer_fn=lambda: tf.train.GradientDescentOptimizer(learning_rate),
        adv_config=adv_config)
    input_fn = single_batch_input_fn({FEATURE_NAME: x0}, y0)
    adv_est.train(input_fn=input_fn, steps=1)

    # Computes the gradients on original and adversarial examples.
    orig_pred = np.dot(x0, weight) + bias  # [9.0]
    orig_grad_w = 2 * (orig_pred - y0) * x0.T  # [[2.0], [2.0]]
    orig_grad_b = 2 * (orig_pred - y0).reshape((1,))  # [2.0]
    grad_x = 2 * (orig_pred - y0) * weight.T  # [[8.0, 6.0]]
    perturbation = adv_step_size * grad_x / np.linalg.norm(grad_x)
    x_adv = x0 + perturbation  # [[1.08, 1.06]]
    adv_pred = np.dot(x_adv, weight) + bias  # [9.5]
    adv_grad_w = 2 * (adv_pred - y0) * x_adv.T  # [[3.24], [3.18]]
    adv_grad_b = 2 * (adv_pred - y0).reshape((1,))  # [3.0]

    new_bias = bias - learning_rate * (orig_grad_b + adv_grad_b)
    new_weight = weight - learning_rate * (orig_grad_w + adv_grad_w)
    self.assertAllClose(new_bias, adv_est.get_variable_value(BIAS_VARIABLE))
    self.assertAllClose(new_weight, adv_est.get_variable_value(WEIGHT_VARIABLE))


if __name__ == '__main__':
  tf.test.main()
