# Copyright 2021 Google LLC
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
"""Neural Clustering Process models with concrete neural network layers."""

from neural_structured_learning.research.neural_clustering.models.ncp_base import NCPBase
import tensorflow as tf


class NCPWithMLP(NCPBase):
  """Neural Clustering Process (NCP) model with multilayer perceptrons (MLP).

  This class creates a NCP model using MLPs as the neural network functions in
  `NCPBase`.

  Attributes:
    assigned_point_layer_out_dim: Integer, the output dimension of the
      assigned_point_layer sub-network.
    assigned_point_layer_hidden_dims: An array of integers representing the
      dimensions of hidden layers in the assigned_point_layer sub-network.
    unassigned_point_layer_out_dim: Integer, the output dimension of the
      unassigned_point_layer sub-network.
    unassigned_point_layer_hidden_dims: An array of integers representing the
      dimensions of hidden layers in the unassigned_point_layer sub-network.
    cluster_layer_out_dim: Integer, the output dimension of the cluster_layer
      sub-network.
    cluster_layer_hidden_dims: An array of integers representing the dimensions
      of hidden layers in the cluster_layer sub-network.
    logits_layer_hidden_dims: An array of integers representing the dimensions
      of hidden layers in the logits_layer sub-network.
  """

  def __init__(self, assigned_point_layer_out_dim,
               assigned_point_layer_hidden_dims, unassigned_point_layer_out_dim,
               unassigned_point_layer_hidden_dims, cluster_layer_out_dim,
               cluster_layer_hidden_dims, logits_layer_hidden_dims):

    assigned_point_layer = build_mlp(
        out_dim=assigned_point_layer_out_dim,
        hidden_dims=assigned_point_layer_hidden_dims)

    unassigned_point_layer = build_mlp(
        out_dim=unassigned_point_layer_out_dim,
        hidden_dims=unassigned_point_layer_hidden_dims)

    cluster_layer = build_mlp(
        out_dim=cluster_layer_out_dim, hidden_dims=cluster_layer_hidden_dims)

    logits_layer = build_mlp(out_dim=1, hidden_dims=logits_layer_hidden_dims)

    super(NCPWithMLP, self).__init__(
        assigned_point_layer=assigned_point_layer,
        unassigned_point_layer=unassigned_point_layer,
        cluster_layer=cluster_layer,
        logits_layer=logits_layer)


def build_mlp(out_dim, hidden_dims, activation="relu"):
  """Builds a multilayer perceptron (MLP) feedforward neural network.

  Arguments:
    out_dim: Integer, the output layer dimension.
    hidden_dims: An array of integers representing the dimensions of hidden
      layers (excluding the output layer).
    activation: String, the activation function for `tf.keras.layers.Dense`.

  Returns:
    A `tf.keras.Model` model.
  """
  layers = [
      tf.keras.layers.Dense(dim, activation=activation) for dim in hidden_dims
  ]
  layers.append(tf.keras.layers.Dense(out_dim))
  return tf.keras.Sequential(layers)
