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
"""Functionality common in all Graph Agreement Models trainers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import tensorflow as tf


def batch_iterator(inputs,
                   targets=None,
                   batch_size=None,
                   shuffle=False,
                   allow_smaller_batch=False,
                   repeat=True):
  """A generator that provides batches of samples from the provided inputs."""
  if isinstance(inputs, set):
    inputs = np.asarray(list(inputs))
  if not isinstance(inputs, (np.ndarray, list)):
    raise TypeError('Unsupported data type %s encountered.' % type(inputs))
  if targets is not None and not isinstance(targets, (np.ndarray, list)):
    raise TypeError('Unsupported data type %s encountered.' % type(targets))
  num_samples = len(inputs)
  if batch_size is None:
    batch_size = num_samples
  if batch_size > num_samples:
    allow_smaller_batch = True
  keep_going = True
  while keep_going:
    indexes = np.arange(0, num_samples)
    if shuffle:
      np.random.shuffle(indexes)
    shuffled_inputs = inputs[indexes]
    if targets is not None:
      shuffled_targets = targets[indexes]
    for start_index in range(0, num_samples, batch_size):
      if allow_smaller_batch:
        end_index = min(start_index + batch_size, num_samples)
      else:
        end_index = start_index + batch_size
        if end_index > num_samples:
          break
      batch_inputs = shuffled_inputs[start_index:end_index]
      if targets is None:
        yield batch_inputs
      else:
        batch_targets = shuffled_targets[start_index:end_index]
        yield batch_inputs, batch_targets
    if not repeat:
      keep_going = False


def variable_summaries(var):
  """Attach summaries to a tensor (for TensorBoard visualizations)."""
  name = var.name[var.name.rfind('/') + 1:var.name.rfind(':')]
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


class Trainer(object):
  """Abstract class for model trainers."""
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               model,
               abs_loss_chg_tol=1e-10,
               rel_loss_chg_tol=1e-5,
               loss_chg_iter_below_tol=10):
    self.model = model
    self.abs_loss_chg_tol = abs_loss_chg_tol
    self.rel_loss_chg_tol = rel_loss_chg_tol
    self.loss_chg_iter_below_tol = loss_chg_iter_below_tol

  @abc.abstractmethod
  def train(self, data, **kwargs):
    pass

  def check_convergence(self,
                        prev_loss,
                        loss,
                        step,
                        max_iter,
                        iter_below_tol,
                        min_num_iter=0):
    """Checks if training for a model has converged."""
    has_converged = False

    # Check if we have reached the desired loss tolerance.
    loss_diff = abs(prev_loss - loss)
    if loss_diff < self.abs_loss_chg_tol or abs(
        loss_diff / prev_loss) < self.rel_loss_chg_tol:
      iter_below_tol += 1
    else:
      iter_below_tol = 0
    if iter_below_tol >= self.loss_chg_iter_below_tol:
      # print('Loss value converged.')
      has_converged = True

    # Make sure that irrespective of the stop criteria, the minimum required
    # number of iterations is achieved.
    if step < min_num_iter:
      has_converged = False
    else:
      has_converged = True

    # Make sure we don't exceed the max allowed number of iterations.
    if max_iter is not None and step >= max_iter:
      print('Maximum number of iterations reached.')
      has_converged = True
    return has_converged, iter_below_tol
