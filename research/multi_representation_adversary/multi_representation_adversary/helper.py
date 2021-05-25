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

"""Helper functions for training and eval."""

from absl import logging
from multi_representation_adversary import attacks
from multi_representation_adversary import transforms
import tensorflow.compat.v2 as tf


def build_attack_fn(model, transform_name, attack_name, use_logits=True):
  """Builds a compiled tf.function for given transform and attack.

  Args:
    model: A callable which makes inference on input.
    transform_name: The name of the transform to be used before the attack. See
      transforms.TRANSFORM_FUNCTIONS.
    attack_name: The name of the attack to be used. See
      attacks.ATTACK_CONSTRUCTORS.
    use_logits: A Boolean indicating whether the model returns logits or class
      probabilities.

  Returns:
    A compiled tf.function which generates adversarial input on given input and
    label.
  """
  transform_fn = transforms.TRANSFORM_FUNCTIONS[transform_name]
  inv_transform_fn = transforms.INVERSE_TRANSFORM_FUNCTIONS[transform_name]
  attack = attacks.construct_attack(attack_name)

  pgd_model_fn = lambda x: model(inv_transform_fn(x))
  pgd_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=use_logits, reduction=tf.keras.losses.Reduction.NONE)

  @tf.function
  def attack_fn(x, y):
    # transform -> attack -> inverse transform -> clip.
    transformed_x = transform_fn(x)
    adv_x = attack.attack(transformed_x, y, pgd_model_fn, pgd_loss_fn,
                          random_start=True)
    return tf.clip_by_value(inv_transform_fn(adv_x), 0.0, 1.0)

  return attack_fn


def build_train_step_fn(model, optimizer, loss_fn, metrics, attack_fn):
  """Builds a compiled tf.function for training the model for one step.

  Args:
    model: A Keras Model object which makes inference on input.
    optimizer: A Keras Optimizer object to be used for training.
    loss_fn: A callable representing the loss objective for training.
    metrics: A list of Keras Metric objects to be updated during training.
    attack_fn: A callable built by build_attack_fn.

  Returns:
    A compiled tf.function which trains the model for one step.
  """
  @tf.function
  def train_step(x, y):
    adv_x = attack_fn(x, y)
    with tf.GradientTape() as tape:
      logits = model(adv_x, training=True)
      loss_value = loss_fn(y, logits) + sum(model.losses)
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss_value, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    for metric in metrics:
      metric.update_state(y, logits)
    return loss_value

  return train_step


def build_eval_step_fn(model, metrics, attack_fn):
  """Builds a compiled tf.function for evaluating the model for one step.

  Args:
    model: A Keras Model object which makes inference on input.
    metrics: A list of Keras Metric objects to be updated during evaluation.
    attack_fn: A callable built by build_attack_fn.

  Returns:
    A compiled tf.function which evaluates the model for one step.
  """
  @tf.function
  def eval_step(x, y):
    # Forward pass
    adv_x = attack_fn(x, y)
    logits = model(adv_x)
    for metric in metrics:
      metric.update_state(y, logits)
    return logits

  return eval_step


def load_checkpoint(path, model_fn):
  """Loads a model checkpoint from the given path.

  Args:
    path: The path to the model checkpoint.
    model_fn: A callable which builds the model structure.

  Returns:
    A model which weights are restored from the checkpoint.
  """
  logging.info("Loading checkpoint: %s", path)
  model = model_fn()
  ckpt = tf.train.Checkpoint(model=model)
  ckpt.restore(path)
  return model
