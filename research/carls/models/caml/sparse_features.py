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
r"""Sparse feature embedding library using the CAML theory implemented by CARLS.

Please see the README.md file for more details.
"""

import collections
import threading
import typing
from research.carls import dynamic_embedding_config_pb2 as de_config_pb2
from research.carls import dynamic_embedding_ops as de_ops
from six.moves import range
import tensorflow as tf

# Map from feature name to _BagOfSparseFeaturesEmbeddingParams.
_feature_embedding_collections = {}
_lock = threading.Lock()


class BagOfSparseFeatureVariables(
    collections.namedtuple(
        "SparseFeatureVariables",
        ["context_free_vector", "sigma_kernel", "sigma_bias"])):
  """Trainable variables for a bag of sparse features in the CA-BSFE model.

  Attributes:
   context_free_vector: A `tf.Variable` of shape [embedding_dim].
   sigma_kernel: A `tf.Variable` of shape [sigma_dim].
   sigma_bias: A `tf.Variable` of shape [1].
  """


class _BagOfSparseFeaturesEmbeddingParams(object):
  """Params needed for saving the necessary information during export."""

  def __init__(self, embedding_dim: int, sigma_dim: int):
    r"""Constructor of _BagOfSparseFeaturesEmbeddingParams.

    Args:
      embedding_dim: An integer, the embedding dimension for the
        informative part of the input. It is also the embedding dimension for
        the output.
      sigma_dim: An integer, the variable size for computing the \sigma(x)
        function for each input x.
    """
    self.embedding_dim = embedding_dim
    self.sigma_dim = sigma_dim


class SparseFeatureEmbedding(tf.keras.layers.Layer):
  """A Keras Layer for sparse feature embedding based on CAML.

  Example usage:
    feat_embed = SparseFeatureEmbedding(
        de_config, {"first_fea": (10, 1), "second_fea": (5, 1)})

    # emb is a [batch_size, 15] Tensor.
    emb = feat_embed({"first_fea": first_tensor, "second_fea": second_tensor})
  """

  def __init__(self,
               de_config: de_config_pb2.DynamicEmbeddingConfig,
               feature_map: typing.Dict[typing.Text, typing.Tuple[int, int]],
               op_name: typing.Text,
               em_steps: int = 0,
               service_address: typing.Text = "",
               timeout_ms: int = -1):
    """Constructor of SparseFeatureEmbedding.

    Args:
      de_config: A de_config_pb2.DynamicEmbeddingConfig that configures the
        embedding.
      feature_map: A map from feature name to (embedding_dim, sigma_dim) pair
      op_name: A string, along with de_config, to uniquely identifying the
        embedding data in knowledge bank.
      em_steps: An integer, if not zero, specifies the steps for training the
        embedding and sigma function alternatively. It is show in the CAML paper
        that sometimes a better performance can be achieved if we do not train
        both embedding and sigma function simultaneously.
      service_address: A string denoting the address to a KnowledgeBankService.
      timeout_ms: A integer, if positive, denoting the timeout deadline in
        milliseconds when connecting to the KnowledgeBankService.
    """
    super(SparseFeatureEmbedding, self).__init__()
    if not op_name:
      raise ValueError(
          "Must use a non-empty op_name for each SparseFeatureEmbedding.")
    self._feature_map = feature_map
    self._variable_map = {}
    self._de_config = de_config
    self._op_name = op_name
    self._em_steps = em_steps
    self._service_address = service_address
    self._timeout_ms = timeout_ms

  @property
  def variable_map(self):
    return self._variable_map

  def build(self, input_shape):
    for name, (embed_dim, sigma_dim) in self._feature_map.items():
      if sigma_dim > 0:
        vc_vec = self.add_weight(
            name="%s/context_free_vector" % name,
            shape=[embed_dim],
            dtype=tf.float32,
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal)
        sigma_kernel = self.add_weight(
            name="%s/sigma_kernel" % name,
            shape=[sigma_dim],
            dtype=tf.float32,
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal)
        sigma_bias = self.add_weight(
            name="%s/sigma_bias" % name,
            shape=[1],
            dtype=tf.float32,
            trainable=True,
            initializer=tf.keras.initializers.zeros)
        self._variable_map[name] = BagOfSparseFeatureVariables(
            vc_vec, sigma_kernel, sigma_bias)

  def call(self, feature_keys):
    (embedding_concat, self.vc_concat, self.sigma_concat,
     self.input_embedding_map) = self.lookup(feature_keys)
    return embedding_concat

  def lookup(self, feature_keys):
    """Returns the embeddings of given feature keys.

    Args:
      feature_keys: A string `Tensor` of shape [batch_size, None] or a map from
        feature name to a string `Tensor` of shape [batch_size, None].

    Returns:
      A tuple of
      - A float `Tensor` of shape [batch_size, concat_embedding_dim] where
        concat_embedding_dim is the concated embedding of all feature columns.
      - A float `Tensor` of shape [batch_size, concat_vc_embedding_dim].
      - A float `Tensor` of shape [batch_size, concat_sigma_embedding_dim].
      - A map from feature name to a float `Tensor` of shape
        [batch_size, embedding_dim] (2D) or
        [batch_size, max_sequence_length, embedding_dim] (3D) representing the
        input embedding before applying embedding decomposition formula.
    Raises:
      RuntimeError: if feature_map is empty.
    """
    if not self._feature_map:
      raise RuntimeError("Feature map is empty.")
    if not isinstance(feature_keys, dict):
      if len(self._feature_map) != 1:
        raise TypeError(
            "Input a single key tensor but the feature size is not one: %d." %
            len(self._feature_map))
      feature_keys = {list(self._feature_map.keys())[0]: feature_keys}
    embedding_list = []
    vc_list = []
    sigma_list = []
    input_embedding_map = {}
    for name, keys in feature_keys.items():
      if name not in self._feature_map:
        raise RuntimeError("input feature %s is not in feature map." % name)
      embed_dim, sigma_dim = self._feature_map[name]
      input_variables = None
      if name in self._variable_map.keys():
        input_variables = self._variable_map[name]
      embed, vc, sigma, input_embed, variables = embed_single_feature(
          keys,
          self._de_config,
          embed_dim,
          sigma_dim,
          "%s_%s" % (self._op_name, name),
          self._em_steps,
          input_variables,
          service_address=self._service_address,
          timeout_ms=self._timeout_ms)
      if not input_variables:
        self._variable_map[name] = variables
      embedding_list.append(embed)
      input_embedding_map[name] = input_embed
      if vc is not None:
        vc_list.append(vc)
      if sigma is not None:
        sigma_list.append(sigma)

    vc_concat = tf.concat(vc_list, -1) if vc_list else None
    sigma_concat = tf.concat(sigma_list, -1) if sigma_list else None
    embedding_concat = tf.concat(embedding_list, -1)
    return (embedding_concat, vc_concat, sigma_concat, input_embedding_map)


def _get_shape_as_list(tensor):
  """Replaces None with -1 in the tensor shape list representation."""
  shape = tensor.get_shape().as_list()
  for i, v in enumerate(shape):
    if v is None:
      shape[i] = -1
  return shape


def _partitioned_dynamic_embedding_lookup(
    keys: tf.Tensor,
    de_config: de_config_pb2.DynamicEmbeddingConfig,
    embedding_dim: int,
    sigma_dim: int,
    feature_name: typing.Text,
    service_address: typing.Text = "",
    timeout_ms: int = -1):
  """Partitions the embeddings of a given `keys` into from given keys.

  Args:
    keys: A string `Tensor` of shape [batch] or [batch_size, max_length].
    de_config: Proto DynamicEmbeddingConfig that configs the embedding.
    embedding_dim: a non-negative int denoting the dimension for
      embedding.
    sigma_dim: a positive int denoting the dimension for sigma function.
    feature_name: Feature name for the embedding.
    service_address: The address of a knowledge bank service. If empty, the
      value passed from --kbs_address flag will be used instead.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.

  Returns:
    A tuple of `Tensor` of shapes
    - [batch_size, embedding_dim] and [batch_size, sigma_dim], if keys is 2D.
    - [batch_size, max_sequence_length, embedding_dim] and
      [batch_size, max_sequence_length, sigma_dim], if keys is 3D.
  Raises:
    ValueError: if keys' dimension is not 1D or 2D.
  """
  config = de_config_pb2.DynamicEmbeddingConfig()
  config.CopyFrom(de_config)
  config.embedding_dimension = sigma_dim + embedding_dim
  emb = de_ops.dynamic_embedding_lookup(
      keys,
      config,
      feature_name,
      service_address=service_address,
      timeout_ms=timeout_ms)
  if sigma_dim == 0:
    return emb, None

  # Decompose the embedding into a sigma part and an embedding part.
  ori_shape = _get_shape_as_list(emb)
  if len(ori_shape) == 2:
    sigma_emb = tf.slice(emb, [0, 0], [-1, sigma_dim])
    embedding = tf.slice(emb, [0, sigma_dim], [-1, -1])
  elif len(ori_shape) == 3:
    sigma_emb = tf.slice(emb, [0, 0, 0], [-1, -1, sigma_dim])
    embedding = tf.slice(emb, [0, 0, sigma_dim], [-1, -1, -1])
  else:
    raise ValueError("Unsupported input shape: %d" % ori_shape)
  return embedding, sigma_emb


def embed_single_feature(keys: tf.Tensor,
                         de_config: de_config_pb2.DynamicEmbeddingConfig,
                         embedding_dim: int,
                         sigma_dim: int,
                         feature_name: typing.Text,
                         em_steps: int = 0,
                         variables: BagOfSparseFeatureVariables = None,
                         service_address: typing.Text = "",
                         timeout_ms: int = -1):
  """Embeds a single feature based on the embedding decomposition formula.

  Args:
    keys: A string `Tensor` of shape [batch_size] for single key batch or
      [batch_size, None] for multiple keys batch. Only non-empty strings are
      considered as valid key.
    de_config: Proto DynamicEmbeddingConfig that configs the embedding.
    embedding_dim: a positive int denoting the dimension for embedding.
    sigma_dim: a non-negative int denoting the dimension for sigma
      function. If sigma_dim = 0, no weighted composition is applied to
      the input, simply an average of the embeddings for each feature is
      returned.
    feature_name: Feature name for the embedding.
    em_steps: number of training steps in each iteration of optimizing sigma and
      embedding alternatively, which is done by not updating the gradients of
      the other. If em_steps <= 0, both sigma and embeddings are optimized
      simultaneously. A proper em_steps can help significantly reduce the
      generalization error.
    variables: a `BagOfSparseFeatureVariables` denoting the variables used for
      computing the CA-BSFE embedding. If None, creates new variables.
    service_address: The address of a knowledge bank service. If empty, the
      value passed from --kbs_address flag will be used instead.
    timeout_ms: Timeout millseconds for the connection. If negative, never
      timout.

  Returns:
    A tuple of
    - embedding: A `Tensor` of shape [batch_size, embedding_dim]
      representing composited embedding vector.
    - vc: A `Tensor` of shape [embedding_dim] representing the
      context vector.
    - sigma: A `Tensor` of shape [batch_size] representing the context free
      probability.
    - input_embedding: A `Tensor` of shape [batch_size, embedding_dim]
      (2D) or [batch_size, max_sequence_length, embedding_dim] (3D)
      representing the input embedding.
    - variables: A list of tf.Variable defined in this function.
  Raises:
    TypeError: If de_config is not an instance of DynamicEmbeddingConfig.
    ValueError: If feature_name is not specified, or sigma_dim < 0 or
      embedding_dim <= 0.
  """
  if not isinstance(de_config, de_config_pb2.DynamicEmbeddingConfig):
    raise TypeError("de_config must be an instance of DynamicEmbeddingConfig.")
  if sigma_dim < 0:
    raise ValueError("Invalid sigma_dim: %d" % sigma_dim)
  if embedding_dim <= 0:
    raise ValueError("Invalid embedding_dim: %d" % embedding_dim)
  if not feature_name:
    raise ValueError("Must specify a valid feature_name.")
  # A single key batch is a [batch_size] input.
  if not isinstance(keys, tf.Tensor):
    keys = tf.convert_to_tensor(keys)
  is_single_key_batch = (len(keys.get_shape().as_list()) == 1)

  # Add to global collection of feature embeddings for export.
  bsfe_params = _BagOfSparseFeaturesEmbeddingParams(embedding_dim,
                                                    sigma_dim)
  with _lock:
    _feature_embedding_collections[feature_name] = bsfe_params

  # Case One: the simplest case when input is a batch of single keys like
  # ['a', 'b', 'c']. Just returns dynamic embedding lookup for each key.
  if sigma_dim == 0 and is_single_key_batch:
    embedding, _ = _partitioned_dynamic_embedding_lookup(
        keys,
        de_config,
        embedding_dim,
        sigma_dim,
        feature_name,
        service_address=service_address,
        timeout_ms=timeout_ms)
    return embedding, None, None, embedding, None

  # Define context vector and sigma function parameters.
  if sigma_dim > 0:
    if variables:
      vc = variables.context_free_vector
      sigma_kernel = variables.sigma_kernel
      sigma_bias = variables.sigma_bias
    else:
      vc = tf.Variable(
          tf.random.normal([embedding_dim]), name="%s_vc" % feature_name)
      sigma_kernel = tf.Variable(
          tf.random.normal([sigma_dim]),
          name="%s_sigma_kernal" % feature_name)
      sigma_bias = tf.Variable([0.0], name="%s_sigma_bias" % feature_name)

  input_embedding, sigma_emb = _partitioned_dynamic_embedding_lookup(
      keys,
      de_config,
      embedding_dim,
      sigma_dim,
      feature_name,
      service_address=service_address,
      timeout_ms=timeout_ms)

  # Allows sigma() and embedding be trained alternatively (every `em_steps`)
  # rather than simultaneously.
  global_step = tf.compat.v1.train.get_global_step()
  if global_step is not None and sigma_dim > 0 and em_steps > 0:
    should_update_embedding = tf.equal(tf.mod(global_step / em_steps, 2), 0)
    should_update_sigma = tf.equal(tf.mod(global_step / em_steps, 2), 1)

    # pylint: disable=g-long-lambda
    input_embedding = tf.cond(should_update_embedding, lambda: input_embedding,
                              lambda: tf.stop_gradient(input_embedding))
    sigma_emb = tf.cond(should_update_sigma, lambda: sigma_emb,
                        lambda: tf.stop_gradient(sigma_emb))
    # Without the following two statements also works.
    sigma_kernel = tf.cond(should_update_sigma, lambda: sigma_kernel,
                           lambda: tf.stop_gradient(sigma_kernel))
    sigma_bias = tf.cond(should_update_sigma, lambda: sigma_bias,
                         lambda: tf.stop_gradient(sigma_bias))
    # pylint: enable=g-long-lambda

  # `variables` comes from either input or local definition.
  if sigma_dim > 0 and variables is None:
    variables = BagOfSparseFeatureVariables(vc, sigma_kernel, sigma_bias)

  # Case Two: input is a batch of keys but sigma embedding is non-zero.
  # It reduces to computing the embedding decomposition for each input, i.e.,
  # [emb(x) = sigma(x) * vc + (1 - sigma(x)) * emb_i(x) for x in keys].
  if is_single_key_batch:  # and sigma_dim != 0
    # shape [batch_size, sigma_dim]
    sigma = tf.matmul(sigma_emb, tf.expand_dims(sigma_kernel, [-1]))
    sigma = tf.sigmoid(tf.nn.bias_add(sigma, sigma_bias))
    # shape [batch_size, embedding_dim]
    embedding = tf.reshape(input_embedding, [-1, embedding_dim])
    embedding = sigma * vc + (1 - sigma) * embedding
    return embedding, vc, sigma, input_embedding, variables

  # Case Three: the rank of input keys > 1, e.g., [['a', 'b'], ['c', '']].
  # The bag of sparse features embedding for each example is computed.
  shape_list = _get_shape_as_list(keys)
  shape_list.append(embedding_dim)
  if sigma_dim > 0:
    sigma_emb = tf.reshape(sigma_emb, [-1, sigma_dim])
    sigma = tf.matmul(sigma_emb, tf.expand_dims(sigma_kernel, [-1]))
    sigma = tf.sigmoid(tf.nn.bias_add(sigma, sigma_bias))
    embedding = tf.reshape(input_embedding, [-1, embedding_dim])
    embedding = (sigma * vc + (1 - sigma) * embedding)
    embedding = tf.reshape(embedding, shape_list)
    sigma = tf.reshape(sigma, shape_list[:-1])
  else:
    embedding = input_embedding
    sigma = None
    vc = None

  # Only computes the average embeddings over the non-empty features.
  mask = tf.cast(tf.not_equal(keys, tf.zeros_like(keys)), dtype=tf.float32)
  mask = tf.reduce_sum(mask, -1)
  mask = tf.where(tf.equal(mask, tf.zeros_like(mask)), tf.ones_like(mask), mask)
  mask = 1 / mask
  mask = tf.expand_dims(mask, -1)  # [batch_size, 1]
  tile_shape = _get_shape_as_list(keys)
  for i in range(len(tile_shape)):
    tile_shape[i] = 1
  tile_shape[-1] = embedding_dim
  mask = tf.tile(mask, tile_shape)
  embedding = tf.reduce_sum(embedding, -2)
  embedding *= mask
  return (embedding, vc, sigma, input_embedding,
          variables if sigma_dim > 0 else None)
