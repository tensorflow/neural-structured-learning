from __future__ import absolute_import, division, print_function

import tensorflow as tf

from gam.models.models_base import Model
from gam.models.models_base import glorot

# Global unique layer ID dictionary for layer name assignment.
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
  """Helper function, assigns unique layer IDs."""
  if layer_name not in _LAYER_UIDS:
    _LAYER_UIDS[layer_name] = 1
    return 1
  else:
    _LAYER_UIDS[layer_name] += 1
    return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
  """Dropout for sparse tensors."""
  random_tensor = keep_prob
  random_tensor += tf.random_uniform(noise_shape)
  dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
  pre_out = tf.sparse_retain(x, dropout_mask)
  return tf.SparseTensor(
    indices=pre_out.indices,
    values=pre_out.values / keep_prob,
    dense_shape=pre_out.dense_shape)


def dot(x, y, sparse=False):
  """Wrapper for tf.matmul (sparse vs dense)."""
  if sparse:
    res = tf.sparse_tensor_dense_matmul(x, y)
  else:
    res = tf.matmul(x, y)
  return res


class GCN(Model):
  def __init__(self,
               input_dim,
               output_dim,
               hidden,
               dropout,
               data,
               aggregation=None,
               hidden_aggregation=(),
               activation=tf.nn.leaky_relu,
               is_binary_classification=False,
               name='GCN'):
    super(GCN, self).__init__(
      aggregation=aggregation,
      hidden_aggregation=hidden_aggregation,
      activation=activation)

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.num_supports = 1
    self.hidden = hidden
    self.dropout = dropout
    self.name = name
    self.is_binary_classification = is_binary_classification

    self.num_features_nonzero = data.features[1].shape

    # Create some Tensorflow placeholders that are specific to GCN.
    self.support_op = tf.sparse_placeholder(tf.float32, name='support')
    self.features_op = tf.sparse_placeholder(
        tf.float32,
        shape=tf.constant((data.num_nodes, data.num_features), dtype=tf.int64),
        name='features')
    # Save the data required to fill in these placeholders. We don't add them
    # directly in the graph as constants in order to avoid saving large
    # checkpoints.
    self.support = data.support
    self.features = data.features

  def get_encoding_and_params(self, inputs, is_train, **unused_kwargs):
    """Creates the model hidden representations and prediction ops.

    For this model, the hidden representation is the last layer of the MLP,
    before the logit computation. The predictions are unnormalized logits.

    Args:
      inputs: A tensor containing the model inputs. The first dimension is the
        batch size.
      **unused_kwargs: Other unused keyword arguments.

    Returns:
      encoding: A tensor containing an encoded batch of samples. The first
        dimension corresponds to the batch size.
      all_vars: A dictionary mapping from variable name to TensorFlow op
        containing all variables used in this model.
      reg_params: A dictionary mapping from a variable name to a Tensor of
        parameters which will be used for regularization.
    """
    # Build layers.
    with tf.variable_scope(self.name):
      if isinstance(inputs, (tuple, list)):
        with tf.variable_scope('encoding'):
          hidden1, reg_params = self._construct_encoding(inputs[0], is_train)
        with tf.variable_scope('encoding', reuse=True):
          hidden2, _ = self._construct_encoding(inputs[1], is_train)
        hidden = self._aggregate((hidden1, hidden2))
      else:
        with tf.variable_scope('encoding'):
          hidden, reg_params = self._construct_encoding(inputs, is_train)

      # Store model variables for easy access.
      variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=tf.get_default_graph().get_name_scope())
      all_vars = {var.name: var for var in variables}

    return hidden, all_vars, reg_params

  def _construct_encoding(self, inputs, is_train):
      """Create weight variables."""
      dropout = (
          tf.constant(self.dropout, tf.float32) * tf.cast(is_train, tf.float32))

      layer_1 = GraphConvolution(
          input_dim=self.input_dim,
          output_dim=self.hidden,
          act=tf.nn.relu,
          dropout=dropout,
          sparse_inputs=True,
          num_features_nonzero=self.num_features_nonzero,
          support=self.support,
          name='GraphConvolution1')
      encoding = layer_1(inputs)
      reg_params = layer_1.vars

      return encoding, reg_params


  def get_predictions_and_params(self, encoding, is_train, **kwargs):
    """Creates the model prediction op.

    For this model, the hidden representation is the last layer of the MLP,
    before the logit computation. The predictions are unnormalized logits.

    Args:
      encoding: A tensor containing the model inputs. The first dimension is the
        batch size.
      is_train: A placeholder representing a boolean value that specifies if
        this model will be used for training or for test.
      **kwargs: Other keyword arguments.

    Returns:
      predictions: A tensor of logits. For multiclass classification its
        shape is (num_samples, num_classes), where the second dimension contains
        a logit per class. For binary classification, its shape is
        (num_samples,), where each element is the probability of class 1 for
        that sample.
      all_vars: A dictionary mapping from variable name to TensorFlow op
        containing all variables used in this model.
      reg_params: A dictionary mapping from a variable name to a Tensor of
        parameters which will be used for regularization.
    """
    reg_params = {}

    # Build layers.
    with tf.variable_scope(self.name + '/prediction'):
      dropout = (
          tf.constant(self.dropout, tf.float32) * tf.cast(is_train, tf.float32))

      layer_2 = GraphConvolution(
        input_dim=self.hidden,
        output_dim=self.output_dim,
        act=lambda x: x,
        dropout=dropout,
        num_features_nonzero=self.num_features_nonzero,
        support=self.support,
        name='GraphConvolution2')
      predictions = layer_2(encoding)

      if self.is_binary_classification:
          predictions = predictions[:, 0]

      # Store model variables for easy access.
      variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=tf.get_default_graph().get_name_scope())
      all_vars = {var.name: var for var in variables}

    return predictions, all_vars, reg_params


  def get_loss(self,
               predictions,
               targets,
               input_indices,
               name_scope='loss',
               reg_params=None,
               **kwargs):
    """Returns a loss between the provided targets and predictions.

    For binary classification, this loss is sigmoid cross entropy. For
    multi-class classification, it is softmax cross entropy.
    A weight decay loss is also added to the parameters passed in reg_params.

    Arguments:
      predictions: A tensor of predictions. For multiclass classification its
        shape is (num_samples, num_classes), where the second dimension contains
        a logit per class. For binary classification, its shape is
        (num_samples,), where each element is the probability of class 1 for
        that sample.
      targets: A tensor of targets of shape (num_samples,), where each row
        contains the label index of the corresponding sample.
      name_scope: A string containing the name scope used in TensorFlow.
      reg_params: A dictonary of parameters, mapping from name to parameter, for
        the variables to be included in the weight decay loss. If None, no
        weight decay is applied.
      **kwargs: Keyword arguments, potentially containing the weight of the
        regularization term, passed under the name `weight_decay`. If this is
        not provided, it defaults to 0.0.

    Returns:
      loss: The cummulated loss value.
    """
    reg_params = reg_params if reg_params is not None else {}
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else None

    with tf.name_scope(name_scope):
      selected_predictions = tf.gather(predictions, input_indices)
      selected_targets = tf.gather(targets, input_indices)

      # Cross entropy error.
      if self.is_binary_classification:
        loss = tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(
            labels=selected_targets, logits=selected_predictions))
      else:
        loss = tf.losses.softmax_cross_entropy(
          selected_targets, selected_predictions)
      # Weight decay loss.
      if weight_decay is not None:
        for var in reg_params.values():
          loss = loss + weight_decay * tf.nn.l2_loss(var)
    return loss

  def normalize_predictions(self, predictions):
    """Converts predictions to probabilities.

    Arguments:
      predictions: A tensor of logits. For multiclass classification its shape
        is (num_samples, num_classes), where the second dimension contains a
        logit per class. For binary classification, its shape is (num_samples,),
        where each element is the probability of class 1 for that sample.

    Returns:
      A tensor of the same shape as predictions, with values between [0, 1]
    representing probabilities.
    """
    if self.is_binary_classification:
      return tf.nn.sigmoid(predictions)
    return tf.nn.softmax(predictions, axis=-1)

  def add_to_feed_dict(self):
    return {
        self.support_op: self.support,
        self.features_op: self.features}

class GraphConvolution(object):
  """Graph convolution layer."""
  def __init__(self, input_dim, output_dim, support, num_features_nonzero,
               dropout=0., sparse_inputs=False, act=tf.nn.relu, bias=False,
               featureless=False, name=None):
    if not name:
      layer = self.__class__.__name__.lower()
      name = layer + '_' + str(get_layer_uid(layer))
    self.name = name
    self.vars = {}
    self.dropout = dropout
    self.act = act
    self.support = support
    self.sparse_inputs = sparse_inputs
    self.featureless = featureless
    self.bias = bias

    # helper variable for sparse dropout
    self.num_features_nonzero = num_features_nonzero

    with tf.variable_scope(self.name + '_vars'):
      self.vars['weights'] = tf.get_variable(
        name='weights', initializer=glorot([input_dim, output_dim]))
      if self.bias:
        self.vars['bias'] = tf.get_variable(
          name='bias', initializer=tf.zeros(shape=[output_dim]))

  def __call__(self, inputs):
    with tf.name_scope(self.name):
      outputs = self._call(inputs)
      return outputs

  def _call(self, inputs):
    x = inputs

    # Dropout.
    if self.sparse_inputs:
      x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
    else:
      x = tf.nn.dropout(x, 1 - self.dropout)

    # Convolve.
    if not self.featureless:
      pre_sup = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
    else:
      pre_sup = self.vars['weights']
    support = dot(self.support, pre_sup, sparse=True)
    output = support

    # Bias.
    if self.bias:
      output += self.vars['bias']

    return self.act(output)
