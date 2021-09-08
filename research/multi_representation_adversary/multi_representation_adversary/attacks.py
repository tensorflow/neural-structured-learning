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

"""Gradient-based attacks in various norms.

Attacks may be applied in any representation space.
"""
from absl import logging
import gin
import neural_structured_learning as nsl
from multi_representation_adversary.cleverhans_adapter import sparse_l1_descent
from multi_representation_adversary.cleverhans_adapter import utils_tf
import numpy as np
import tensorflow.compat.v2 as tf


class Attack(object):
  """Abstract base class for an attack."""

  def attack(self, x, y, model, loss, random_start=False):
    """Performs an attack to the model on the given example (x, y).

    Args:
      x: A batched tensor. Input features.
      y: A batched tensor. Input labels.
      model: A callable. The model to be attacked.
      loss: A callable. The loss function.
      random_start: A Boolean. Whether to start the attack from a random point
        inside the epsilon ball.
    """
    raise NotImplementedError


class NSLAttack(Attack):
  """Attacks using the Neural Structured Learning library."""

  def __init__(self, adv_config):
    """Constructs an Attack object with the NSL library.

    Args:
      adv_config: An instance of `nsl.configs.AdvNeighborConfig`.
    """
    self.config = adv_config
    # Handle iterations separately to have the correct random_start behavior.
    self.num_iter = self.config.pgd_iterations
    self.config.pgd_iterations = 1
    logging.info("NSL CONFIG CREATION: %s", self.__dict__)

  def attack(self, x, y, model, loss, random_start=False):
    """See base class."""
    if self.config.adv_grad_norm == nsl.configs.NormType.INFINITY:
      order = np.inf
    elif self.config.adv_grad_norm == nsl.configs.NormType.L2:
      order = 2
    else:
      # Not used in the experiments.
      raise ValueError("adv_grad_norm is not \"INFINITY\" or \"L2\" NormType.")

    adv_x0 = x
    if random_start:
      adv_x0 += utils_tf.random_lp_vector(
          tf.shape(x),
          ord=order,
          eps=tf.cast(self.config.pgd_epsilon, tf.float32),
          dtype=adv_x0.dtype)

    def pgd_step(i, adv_xi):
      """Do a projected gradient step for adversarial examples."""
      with tf.GradientTape(watch_accessed_variables=False) as adv_tape:
        adv_tape.watch(adv_xi)
        adv_losses = loss(y, model(adv_xi))

      adv_xi, _ = nsl.lib.gen_adv_neighbor(
          adv_xi,
          adv_losses,
          self.config,
          gradient_tape=adv_tape,
          pgd_model_fn=model,
          pgd_loss_fn=loss,
          pgd_labels=y)

      eta = adv_xi - x
      eta = utils_tf.clip_eta(eta, ord=order, eps=self.config.pgd_epsilon)
      adv_xi = x + eta
      return i + 1, adv_xi

    _, adv_x = tf.while_loop(
        lambda i, _: tf.less(i, self.num_iter),
        pgd_step, (tf.zeros([]), adv_x0),
        back_prop=True,
        maximum_iterations=self.num_iter)
    return adv_x


class SparseL1Attack(Attack):
  """A wrapper for the SparseL1Attack from Cleverhans."""

  def __init__(self, num_iter, epsilon, step_size, q, num_classes):
    """Constructs an Attack object wrapping the SparseL1Attack from Cleverhans.

    Args:
      num_iter: Number of iterations to run the attack.
      epsilon: The epsilon ball to clip to.
      step_size: The step size for the attack.
      q: Sparsity of the gradient update step in percent. Gradient values larger
        than this percentile are retained.
      num_classes: Number of classes in the label space.
    """
    self.q = q
    self.eps = epsilon
    self.eps_iter = step_size
    self.nb_iter = num_iter
    self.num_classes = num_classes
    logging.info("L1 CONFIG CREATION: %s", self.__dict__)

  def attack(self, x, y, model, loss, random_start=False):
    """See base class."""
    y_onehot = tf.reshape(
        tf.one_hot(y, self.num_classes), [-1, self.num_classes])
    adv_x = x
    if random_start:
      perturb = utils_tf.random_lp_vector(
          tf.shape(adv_x),
          ord=1,
          eps=tf.cast(self.eps, tf.float32),
          dtype=adv_x.dtype)
      adv_x = adv_x + perturb

    @tf.function
    def pgd_step(i, adv_x):
      """Do a projected gradient step for adversarial examples."""
      adv_x = sparse_l1_descent.sparse_l1_descent(
          adv_x,
          model(adv_x),
          y=y_onehot,
          eps=self.eps_iter,
          q=self.q,
          clip_min=None,
          clip_max=None,
          sanity_checks=True)

      eta = adv_x - x
      eta = utils_tf.clip_eta(eta, ord=1, eps=self.eps)
      adv_x = x + eta

      return i + 1, adv_x

    _, adv_x = tf.while_loop(
        lambda i, _: tf.less(i, self.nb_iter),
        pgd_step, (tf.zeros([]), adv_x),
        back_prop=True,
        maximum_iterations=self.nb_iter)
    return adv_x


class NoAttack(Attack):
  """An Attack class represents no attack at all."""

  def attack(self, x, y, model, loss, random_start=False):
    """See base class."""
    return x


class UnionAttack(Attack):
  """Composite Attack class with 1+ base attacks and 1+ restarts.

  For each example, this attack evaluates each base attack with N restarts and
  take the one with the highest loss.
  """

  def __init__(self, base_attack_names, restart):
    """Constructs an Attack object wrapping a base attack with restarts.

    Args:
      base_attack_names: List of base attacks, any of ('l1', 'l2', 'linf').
      restart: Number of restarts for each attack.
    """
    self.restart = restart
    self.attacks = list(map(construct_attack, base_attack_names))
    logging.info("UNION CONFIG CREATION: %s", self.__dict__)

  def attack(self, x, y, model, loss, random_start=False):
    """See base class."""
    batch_shape = [x.shape[0]] + [1] * (x.shape.rank - 1)
    adv_x = x
    adv_loss = tf.zeros(batch_shape)
    for attack in self.attacks:
      for _ in range(self.restart):
        adv_x2 = attack.attack(x, y, model, loss, random_start)
        adv_loss2 = tf.reshape(loss(y, model(adv_x2)), batch_shape)
        comparison = adv_loss < adv_loss2
        adv_x = tf.where(comparison, adv_x2, adv_x)
        adv_loss = tf.where(comparison, adv_loss2, adv_loss)
    return adv_x


class SingleRotationAttack(Attack):
  """Single random rotation attack.

  Reference:
  Engstrom et al. Exploring the Landscape of Spatial Robustness.
  https://arxiv.org/pdf/1712.02779.pdf
  """

  def __init__(self, max_degree):
    self.max_degree = max_degree
    logging.info("ROTATION CONFIG CREATION: %s", self.__dict__)
    self.rotate = tf.keras.layers.experimental.preprocessing.RandomRotation(
        factor=max_degree / 360., fill_mode="constant")

  def attack(self, x, y, model, loss, random_start=False):
    """See base class."""
    return self.rotate(x)


@gin.configurable
def linf_config(num_iter=1, epsilon=(8 / 255), step_size=(2 / 255)):
  adv_config = nsl.configs.AdvNeighborConfig(
      adv_grad_norm="infinity",
      pgd_iterations=num_iter,
      adv_step_size=step_size,
      clip_value_min=None,
      clip_value_max=None,
      pgd_epsilon=epsilon)
  return {"adv_config": adv_config}


@gin.configurable
def l2_config(num_iter=1, epsilon=(8 / 255), step_size=(2 / 255)):
  adv_config = nsl.configs.AdvNeighborConfig(
      adv_grad_norm="l2",
      pgd_iterations=num_iter,
      adv_step_size=step_size,
      clip_value_min=None,
      clip_value_max=None,
      pgd_epsilon=epsilon)
  return {"adv_config": adv_config}


@gin.configurable
def l1_config(num_iter=10,
              epsilon=10,
              step_size=1,
              percentile=99,
              num_classes=10):
  return {
      "num_iter": num_iter,
      "epsilon": epsilon,
      "step_size": step_size,
      "q": percentile,
      "num_classes": num_classes,
  }


@gin.configurable
def union_config(base_attack_names, restart=1):
  return {
      "base_attack_names": base_attack_names,
      "restart": restart,
  }


@gin.configurable
def single_rotation_config(max_degree=30.):
  return {"max_degree": max_degree}


@gin.configurable
def union_rotation_config(num_trials=10):
  return {"base_attack_names": ["single_rotation"], "restart": num_trials}


# Attacks cannot be instantiated at global namespace because the *_config()
# calls depend on gin for the correct default parameters, which is initialized
# in main() after the creation of global variables.
ATTACK_CONSTRUCTORS = {
    "linf": lambda: NSLAttack(**linf_config()),
    "l2": lambda: NSLAttack(**l2_config()),
    "l1": lambda: SparseL1Attack(**l1_config()),
    "none": NoAttack,

    # Composite attack. Num of restarts is set in union_config.
    "linf_restart": lambda: UnionAttack(**union_config(["linf"])),
    "l2_restart": lambda: UnionAttack(**union_config(["l2"])),
    "l1_restart": lambda: UnionAttack(**union_config(["l1"])),

    # Rotation attack.
    "single_rotation": lambda: SingleRotationAttack(**single_rotation_config()),
    "union_rotation": lambda: UnionAttack(**union_rotation_config()),
}


def construct_attack(name):
  return ATTACK_CONSTRUCTORS[name]()
