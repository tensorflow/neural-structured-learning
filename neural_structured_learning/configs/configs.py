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
"""Classes for configuring modules in Neural Structured Learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
import enum
import tensorflow as tf


class NormType(enum.Enum):
  """Types of norms."""
  L1 = 'l1'
  L2 = 'l2'
  INFINITY = 'infinity'

  @classmethod
  def all(cls):
    return list(cls)


@attr.s
class AdvNeighborConfig(object):
  """AdvNeighborConfig contains configs for generating adversarial neighbors.

  Attributes:
    feature_mask: mask (w/ 0-1 values) applied on gradient. The shape should be
      the same as (or broadcastable to) input features. If set to None, no
      feature mask will be applied.
    adv_step_size: step size to find the adversarial sample. Default set to
      0.001.
    adv_grad_norm: type of tensor norm to normalize the gradient. Input will be
      converted to `NormType` when applicable (e.g., 'l2' -> NormType.L2).
      Default set to L2 norm.
  """
  feature_mask = attr.ib(default=None)
  adv_step_size = attr.ib(default=0.001)
  adv_grad_norm = attr.ib(converter=NormType, default='l2')


@attr.s
class AdvRegConfig(object):
  """AdvRegConfig contains configs for adversarial regularization.

  Attributes:
    multiplier: multiplier to adversarial regularization loss. Default set to
      0.2.
    adv_neighbor_config: an AdvNeighborConfig object for generating adversarial
      neighbor examples.
  """
  multiplier = attr.ib(default=0.2)
  adv_neighbor_config = attr.ib(default=AdvNeighborConfig())


def make_adv_reg_config(
    multiplier=attr.fields(AdvRegConfig).multiplier.default,
    feature_mask=attr.fields(AdvNeighborConfig).feature_mask.default,
    adv_step_size=attr.fields(AdvNeighborConfig).adv_step_size.default,
    adv_grad_norm=attr.fields(AdvNeighborConfig).adv_grad_norm.default):
  """Creates AdvRegConfig object.

  Args:
    multiplier: multiplier to adversarial regularization loss. Default set to
      0.2.
    feature_mask: mask (w/ 0-1 values) applied on gradient. The shape should be
      the same as (or broadcastable to) input features. If set to None, no
      feature mask will be applied.
    adv_step_size: step size to find the adversarial sample. Default set to
      0.001.
    adv_grad_norm: type of tensor norm to normalize the gradient. Input will be
      converted to `NormType` when applicable (e.g., 'l2' -> NormType.L2).
      Default set to L2 norm.

  Returns:
    An AdvRegConfig object.
  """
  return AdvRegConfig(
      multiplier=multiplier,
      adv_neighbor_config=AdvNeighborConfig(
          feature_mask=feature_mask,
          adv_step_size=adv_step_size,
          adv_grad_norm=adv_grad_norm))


class AdvTargetType(enum.Enum):
  """Types of adversarial targeting."""
  SECOND = 'second'
  LEAST = 'least'
  RANDOM = 'random'
  GROUND_TRUTH = 'ground_truth'

  @classmethod
  def all(cls):
    return list(cls)


@attr.s
class AdvTargetConfig(object):
  """AdvTargetConfig contains configs for selecting targets to be attacked.

  Attributes:
    target_method: type of adversarial targeting method. The value needs to be
      one of the enums from AdvTargetType (e.g., AdvTargetType.LEAST).
    random_seed: a Python integer as seed in 'random_uniform' op.
  """
  target_method = attr.ib(default=AdvTargetType.GROUND_TRUTH)
  random_seed = attr.ib(default=0.0)


class TransformType(enum.Enum):
  """Types of nonlinear functions to be applied ."""
  SOFTMAX = 'softmax'
  NONE = 'none'


class DistanceType(enum.Enum):
  """Types of distance."""
  L1 = 'l1'
  L2 = 'l2'
  COSINE = 'cosine'
  JENSEN_SHANNON_DIVERGENCE = 'jensen_shannon_divergence'
  KL_DIVERGENCE = 'kl_divergence'

  @classmethod
  def all(cls):
    return list(cls)


@attr.s
class DistanceConfig(object):
  """DistanceConfig contains configs for computing distances.

  Attributes:
    distance_type: type of distance function. Input type will be converted to
      'DistanceType' when applicable (e.g., 'l2' -> DistanceType.L2). Default
      set to L2 norm.
    reduction: type of distance reduction. See tf.compat.v1.losses.Reduction for
      details. Default set to tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS.
    sum_over_axis: the distance is the sum over the difference along the axis.
      Default set to None.
    transform_fn: type of transform function to be applied on each side before
      computing the pairwise distance. Input type will be converted to
      'TransformType' when applicable (e.g., 'softmax' ->
      TransformType.SOFTMAX). Default set to 'none'.
  """
  distance_type = attr.ib(converter=DistanceType, default=DistanceType.L2)
  reduction = attr.ib(
      default=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
  sum_over_axis = attr.ib(default=None)
  transform_fn = attr.ib(converter=TransformType, default='none')


class DecayType(enum.Enum):
  """Types of decay."""
  EXPONENTIAL_DECAY = 'exponential_decay'
  INVERSE_TIME_DECAY = 'inverse_time_decay'
  NATURAL_EXP_DECAY = 'natural_exp_decay'

  @classmethod
  def all(cls):
    return list(cls)


@attr.s
class DecayConfig(object):
  """DecayConfig contains configs for computing decayed value.

  Attributes:
    decay_steps: A scalar int32 or int64 Tensor or a Python number. How often to
      apply decay. Must be positive.
    decay_rate: A scalar float32 or float64 Tensor or a Python number. Default
      set to 0.96.
    min_value: minimal acceptable value after applying decay. Default set to 0.0
    decay_type: Type of decay function to apply. Default set to
      DecayType.EXPONENTIAL_DECAY.
  """
  decay_steps = attr.ib()
  decay_rate = attr.ib(default=0.96)
  min_value = attr.ib(default=0.0)
  decay_type = attr.ib(default=DecayType.EXPONENTIAL_DECAY)


class IntegrationType(enum.Enum):
  """Types of integration for multimodal fusion."""
  ADD = 'additive'
  MUL = 'multiplicative'
  TUCKER_DECOMP = 'tucker_decomp'

  @classmethod
  def all(cls):
    return list(cls)


@attr.s
class IntegrationConfig(object):
  """IntegrationConfig contains configs for computing multimodal integration.

  Attributes:
    integration_type: Type of integration function to apply.
    hidden_dims: Integer or a list of Integer, the number of hidden units in the
      fully-connected layer(s) before the output layer.
    activation_fn: Activation function to be applied to.
  """
  integration_type = attr.ib(converter=IntegrationType)
  hidden_dims = attr.ib()
  activation_fn = attr.ib(default=tf.nn.tanh)


@attr.s
class VirtualAdvConfig(object):
  """VirtualAdvConfig contains configs for virtual adversarial training.

  Attributes:
    adv_neighbor_config: an AdvNeighborConfig object for generating virtual
      adversarial examples. Default set to AdvNeighborConfig.
    distance_config: a DistanceConfig object for calculating virtual adversarial
      loss. Default set to DistanceConfig.
    num_approx_steps: number of steps used to approximate the calculation of
      Hessian matrix required for creating virtual adversarial examples. Default
      set to 1.
    approx_difference: the finite difference to approximate the calculation of
      Hessian matrix required for creating virtual adversarial examples. (The
      `xi` in Equation 12 in the paper: https://arxiv.org/pdf/1704.03976.pdf)
        Default set to 1e-6.
  """
  adv_neighbor_config = attr.ib(default=AdvNeighborConfig())
  distance_config = attr.ib(default=DistanceConfig())
  num_approx_steps = attr.ib(default=1)
  approx_difference = attr.ib(default=1e-6)


@attr.s
class GraphNeighborConfig(object):
  """GraphNeighborConfig specifies neighbor attributes for graph regularization.

  Attributes:
    prefix: The prefix in feature names that identifies neighbor-specific
      features. Defaults to 'NL_nbr_'.
    weight_suffix: The suffix in feature names that identifies the neighbor
      weight value. Defaults to '_weight'. Note that neighbor weight features
      will have `prefix` as a prefix and `weight_suffix` as a suffix. For
      example, based on the default values of `prefix` and `weight_suffix`, a
      valid neighbor weight feature is 'NL_nbr_0_weight', where 0 corresponds to
      the first neighbor of the sample.
    max_neighbors: The maximum number of neighbors to be used for graph
      regularization. Defaults to 0, which disables graph regularization. Note
      that this value has to be less than or equal to the actual number of
      neighbors in each sample.
  """
  prefix = attr.ib(default='NL_nbr_')
  weight_suffix = attr.ib(default='_weight')
  max_neighbors = attr.ib(default=0)


@attr.s
class GraphRegConfig(object):
  """GraphRegConfig contains the configuration for graph regularization.

  Attributes:
    neighbor_config: An instance of `GraphNeighborConfig` that describes
      neighbor attributes for graph regularization.
    multiplier: The multiplier or weight factor applied on the graph
      regularization loss term. Defaults to 0.01. This value has to be greater
      than or equal to 0.
    distance_config: An instance of `DistanceConfig` to calculate the graph
      regularization loss term. Defaults to `DistanceConfig()`.
  """
  neighbor_config = attr.ib(default=GraphNeighborConfig())
  multiplier = attr.ib(default=0.01)
  distance_config = attr.ib(default=DistanceConfig())


DEFAULT_DISTANCE_PARAMS = attr.asdict(DistanceConfig())
DEFAULT_ADVERSARIAL_PARAMS = attr.asdict(AdvNeighborConfig())
