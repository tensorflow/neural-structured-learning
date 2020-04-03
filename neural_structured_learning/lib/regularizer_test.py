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

"""Tests for neural_structured_learning.lib.regularizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structured_learning.configs as configs
from neural_structured_learning.lib import regularizer
import numpy as np
import tensorflow as tf


def mock_model_fn(adv_input, is_train, reuse):
  """Mock model_fn that returns input directly."""
  del is_train, reuse
  return adv_input


def mock_loss_fn(onehot_labels, logits):
  """Mock loss_fn that returns MSE loss."""
  return tf.keras.losses.mean_squared_error(onehot_labels, logits)


class RegularizerTest(tf.test.TestCase):
  """Tests regularizer methods."""

  def testAdvRegularizer(self):
    """Tests adv_regularizer returns expected adv_loss."""
    adv_neighbor = np.array([1., 1., 1., 0., 1.])
    target = np.array([1., 1., 1., 1., 1.])

    adv_loss = regularizer.adv_regularizer(
        tf.constant(adv_neighbor), tf.constant(target), mock_model_fn,
        mock_loss_fn)
    actual_loss = self.evaluate(adv_loss)
    self.assertNear(
        actual_loss,
        np.sum((adv_neighbor - target)**2.0) / len(target), 1e-5)

  def testVirtualAdvRegularizer(self):
    """Tests virtual_adv_regularizer returning expected loss."""
    np_input = np.array([[1.0, -1.0]])
    tf_input = tf.constant(np_input)
    np_weights = np.array([[1.0, 5.0], [2.0, 2.0]])
    tf_weights = tf.constant(np_weights)
    # Linear transformation and L2 loss makes the Hessian matrix constant.
    embedding_fn = lambda x: tf.matmul(x, tf_weights)
    step_size = 0.1
    vadv_config = configs.VirtualAdvConfig(
        adv_neighbor_config=configs.AdvNeighborConfig(
            feature_mask=None,
            adv_step_size=step_size,
            adv_grad_norm=configs.NormType.L2),
        distance_config=configs.DistanceConfig(
            distance_type=configs.DistanceType.L2, sum_over_axis=-1),
        num_approx_steps=1,
        approx_difference=1e-3)  # enlarged for numerical stability
    np_seed = np.array([[0.6, 0.8]])
    tf_seed = tf.constant(np_seed)
    vadv_loss = regularizer._virtual_adv_regularizer(tf_input, embedding_fn,
                                                     vadv_config,
                                                     embedding_fn(tf_input),
                                                     tf_seed)

    actual_loss = self.evaluate(vadv_loss)

    hessian = 2 * np.dot(np_weights, np_weights.T)
    approx = np.matmul(np_seed, hessian)
    approx *= step_size / np.linalg.norm(approx, axis=-1, keepdims=True)
    expected_loss = np.linalg.norm(np.matmul(approx, np_weights))**2
    self.assertNear(actual_loss, expected_loss, err=1e-5)

  def testVirtualAdvRegularizerMultiStepApproximation(self):
    """Tests virtual_adv_regularizer with multi-step approximation."""
    np_input = np.array([[0.28, -0.96]])
    tf_input = tf.constant(np_input)
    embedding_fn = lambda x: x
    vadv_config = configs.VirtualAdvConfig(
        adv_neighbor_config=configs.AdvNeighborConfig(
            feature_mask=None,
            adv_step_size=1,
            adv_grad_norm=configs.NormType.L2),
        distance_config=configs.DistanceConfig(
            distance_type=configs.DistanceType.COSINE, sum_over_axis=-1),
        num_approx_steps=20,
        approx_difference=1)
    np_seed = np.array([[0.6, 0.8]])
    tf_seed = tf.constant(np_seed)
    vadv_loss = regularizer._virtual_adv_regularizer(tf_input, embedding_fn,
                                                     vadv_config,
                                                     embedding_fn(tf_input),
                                                     tf_seed)

    actual_loss = self.evaluate(vadv_loss)

    x = np_input
    hessian = np.dot(x, x.T) * np.identity(2) - np.dot(x.T, x)
    hessian /= np.linalg.norm(x)**4
    approx = np.matmul(np_seed, hessian)
    approx /= np.linalg.norm(approx, axis=-1, keepdims=True)
    expected_loss = np.matmul(np.matmul(approx, hessian), np.transpose(approx))
    self.assertNear(actual_loss, expected_loss, err=1e-5)

  def testVirtualAdvRegularizerRandomPerturbation(self):
    """Tests virtual_adv_regularizer with num_approx_steps=0."""
    input_layer = tf.constant([[1.0, -1.0]])
    embedding_fn = lambda x: x
    step_size = 0.1
    vadv_config = configs.VirtualAdvConfig(
        adv_neighbor_config=configs.AdvNeighborConfig(
            feature_mask=None,
            adv_step_size=step_size,
            adv_grad_norm=configs.NormType.L2),
        distance_config=configs.DistanceConfig(
            distance_type=configs.DistanceType.L2, sum_over_axis=-1),
        num_approx_steps=0)
    vadv_loss = regularizer.virtual_adv_regularizer(input_layer, embedding_fn,
                                                    vadv_config)
    actual_loss = self.evaluate(vadv_loss)

    # The identity embedding_fn makes the virtual adversarial loss immune to the
    # direction of the perturbation, only the size matters.
    expected_loss = step_size**2  # square loss
    self.assertNear(actual_loss, expected_loss, err=1e-5)

  def testVirtualAdvRegularizerDefaultConfig(self):
    """Tests virtual_adv_regularizer with default config."""
    input_layer = tf.constant([[1.0, -1.0]])
    embedding_fn = lambda x: x
    vadv_config = configs.VirtualAdvConfig()
    vadv_loss = regularizer.virtual_adv_regularizer(input_layer, embedding_fn,
                                                    vadv_config)
    actual_loss = self.evaluate(vadv_loss)
    self.assertAllGreaterEqual(actual_loss, 0.0)


if __name__ == '__main__':
  tf.test.main()
