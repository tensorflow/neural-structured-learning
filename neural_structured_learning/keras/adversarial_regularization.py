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
"""Incorporates adversarial regularization into a Keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import types

import attr
import neural_structured_learning.configs as nsl_configs
import neural_structured_learning.lib as nsl_lib
import six
import tensorflow as tf
import tensorflow.keras as keras


def adversarial_loss(features,
                     labels,
                     model,
                     loss_fn,
                     sample_weights=None,
                     adv_config=None,
                     predictions=None,
                     labeled_loss=None,
                     gradient_tape=None):
  """Computes the adversarial loss for `model` given `features` and `labels`.

  This utility function adds adversarial perturbations to the input `features`,
  runs the `model` on the perturbed features for predictions, and returns the
  corresponding loss `loss_fn(labels, model(perturbed_features))`. This function
  can be used in a Keras subclassed model and a custom training loop. This can
  also be used freely as a helper function in eager execution mode.

  The adversarial perturbation is based on the gradient of the labeled loss on
  the original input features, i.e. `loss_fn(labels, model(features))`.
  Therefore, this function needs to compute the model's predictions on the input
  features as `model(features)`, and the labeled loss as `loss_fn(labels,
  predictions)`. If predictions or labeled loss have already been computed, they
  can be passed in via the `predictions` and `labeled_loss` arguments in order
  to save computational resources. Note that in eager execution mode,
  `gradient_tape` needs to be set accordingly when passing in `predictions` or
  `labeled_loss`, so that the gradient can be computed correctly.

  Example:
  ```python
  # A linear regression model (for demonstrating the usage only)
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
  loss_fn = tf.keras.losses.MeanSquaredError()
  optimizer = tf.keras.optimizers.SGD()

  # Custom training loop. (The actual training data is omitted for clarity.)
  for x, y in train_dataset:
    with tf.GradientTape() as tape_w:

      # A separate GradientTape is needed for watching the input.
      with tf.GradientTape() as tape_x:
        tape_x.watch(x)

        # Regular forward pass.
        labeled_loss = loss_fn(y, model(x))

      # Calculates the adversarial loss. This will reuse labeled_loss and will
      # consume tape_x.
      adv_loss = nsl.keras.adversarial_loss(
          x, y, model, loss_fn, labeled_loss=labeled_loss, gradient_tape=tape_x)

      # Combines both losses. This could also be a weighted combination.
      total_loss = labeled_loss + adv_loss

    # Regular backward pass.
    gradients = tape_w.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  ```

  Arguments:
    features: Input features, should be a `Tensor` or a collection of `Tensor`
      objects. If it is a collection, the first dimension of all `Tensor`
      objects inside should be the same (i.e. batch size).
    labels: Target labels.
    model: A callable that takes `features` as inputs and computes `predictions`
      as outputs. An example would be a `tf.keras.Model` object.
    loss_fn: A callable which calcualtes labeled loss from `labels`,
      `predictions`, and `sample_weights`. An example would be a
      `tf.keras.losses.Loss` object.
    sample_weights: (optional) A 1-D `Tensor` of weights for the examples, with
      the same length as the first dimension of `features`.
    adv_config: (optional) An `nsl.configs.AdvRegConfig` object for adversarial
      regularization hyperparameters. Use `nsl.configs.make_adv_reg_config` to
      construct one.
    predictions: (optional) Precomputed value of `model(features)`. If set, the
      value will be reused when calculating adversarial regularization. In eager
      mode, the `gradient_tape` has to be set as well.
    labeled_loss: (optional) Precomputed value of `loss_fn(labels,
      model(features))`. If set, the value will be reused when calculating
      adversarial regularization. In eager mode, the `gradient_tape` has to be
      set as well.
    gradient_tape: (optional) A `tf.GradientTape` object watching `features`.

  Returns:
    A `Tensor` for adversarial regularization loss, i.e. labeled loss on
    adversarially perturbed features.
  """

  if adv_config is None:
    adv_config = nsl_configs.AdvRegConfig()

  # Calculates labeled_loss if not provided.
  if labeled_loss is None:
    # Reuses the tape if provided; otherwise creates a new tape.
    gradient_tape = gradient_tape or tf.GradientTape()
    with gradient_tape:
      gradient_tape.watch(tf.nest.flatten(features))
      # Calculates prediction if not provided.
      predictions = predictions if predictions is not None else model(features)
      labeled_loss = loss_fn(labels, predictions, sample_weights)

  adv_input, adv_sample_weights = nsl_lib.gen_adv_neighbor(
      features,
      labeled_loss,
      config=adv_config.adv_neighbor_config,
      gradient_tape=gradient_tape)
  adv_output = model(adv_input)
  if sample_weights is not None:
    adv_sample_weights = tf.math.multiply(sample_weights, adv_sample_weights)
  adv_loss = loss_fn(labels, adv_output, adv_sample_weights)
  return adv_loss


class _LossWrapper(tf.keras.losses.Loss):
  """Wrapper converting a loss function into a `Loss` object.

  This is to reuse logic of sample-weighted loss computation in `Loss` base
  class.

  Attributes:
    loss_fn: Underlying loss function.
    weight: Weight of this loss term in total loss. Should be applied outside
      of this class, e.g. `total_loss += loss.weight * loss(y_true, y_pred)`.
    batch_size_reduction: Whether to perform `SUM_OVER_BATCH_SIZE` reduction.
      This field is set in lieu of having `reduction=SUM_OVER_BATCH_SIZE`,
      because the latter is not supported when using with
      `tf.distribute.Strategy`.
  """

  def __init__(self, loss_fn, name, weight):
    reduction = getattr(loss_fn, 'reduction', None)
    if reduction in (None, tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
                     tf.compat.v2.losses.Reduction.AUTO):
      reduction = tf.losses.Reduction.NONE
      self.batch_size_reduction = True
    else:
      self.batch_size_reduction = False
    super(_LossWrapper, self).__init__(name=name, reduction=reduction)
    self.loss_fn = loss_fn
    self.weight = weight

  def call(self, y_true, y_pred):
    return self.loss_fn(y_true, y_pred)

  def __call__(self, *args, **kwargs):
    loss_value = super(_LossWrapper, self).__call__(*args, **kwargs)
    if self.batch_size_reduction:
      size = tf.cast(tf.size(loss_value), dtype=loss_value.dtype)
      loss_value = tf.math.divide_no_nan(tf.math.reduce_sum(loss_value), size)
    return loss_value

  def _is_sparse_categorical_loss(self):
    return self.loss_fn == keras.losses.sparse_categorical_crossentropy or (
        isinstance(self.loss_fn, keras.losses.SparseCategoricalCrossentropy))

  def _is_binary_classification_loss(self):
    return self.loss_fn in (keras.losses.binary_crossentropy,
                            keras.losses.hinge,
                            keras.losses.squared_hinge) or isinstance(
                                self.loss_fn,
                                (keras.losses.BinaryCrossentropy,
                                 keras.losses.Hinge, keras.losses.SquaredHinge))

  def resolve_metric(self, metric):
    """Resolves potentially ambiguous metric name based on the loss function."""
    # This method is intended for the scenario that a Keras model is compiled
    # with a metric which meaning depends on the learning task. For example,
    # `'accuracy'` may refer to `tf.keras.metrics.binary_accuracy` for binary
    # classification tasks, to `tf.keras.metrics.categorical_accuracy` for
    # multi-class classification tasks with one-hot labels, or to
    # `tf.keras.metrics.sparse_categorical_accuracy` for mult-class
    # classification tasks with index labels. In such scenario the loss
    # function can help deduce the desired metric function since they share the
    # same input `(y_true, y_pred)`.
    # The list of such metrics is here:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training_utils.py#L1018
    if isinstance(metric, keras.metrics.Metric):
      return metric.name
    if metric not in ('accuracy', 'acc', 'crossentropy', 'ce'):
      return metric
    if self._is_binary_classification_loss():
      prefix = 'binary_'
    elif self._is_sparse_categorical_loss():
      prefix = 'sparse_categorical_'
    else:
      prefix = 'categorical_'
    suffix = 'accuracy' if metric in ('accuracy', 'acc') else 'crossentropy'
    return prefix + suffix


def _prepare_loss_fns(loss, output_names):
  """Converts `loss` to a list of per-output loss functions or objects."""
  # losses for multiple outputs indexed by name
  if isinstance(loss, collections.Mapping):
    for name in output_names:
      if name not in loss:
        raise ValueError(
            'Loss for {} not found in `loss` dictionary.'.format(name))
    return [keras.losses.get(loss[name]) for name in output_names]

  # loss for single output, or shared loss fn for multiple outputs
  if isinstance(loss, six.string_types):
    return [keras.losses.get(loss) for _ in output_names]

  # losses for multiple outputs indexed by position
  if isinstance(loss, collections.Sequence):
    if len(loss) != len(output_names):
      raise ValueError('`loss` should have the same number of elements as '
                       'model output')
    return six.moves.map(keras.losses.get, loss)

  # loss for single output, or shared loss fn for multiple outputs
  return [keras.losses.get(loss) for _ in output_names]


def _prepare_loss_weights(loss_weights, output_names):
  """Converts `loss_weights` to a list of float values."""
  if loss_weights is None:
    return [1.0] * len(output_names)

  if isinstance(loss_weights, collections.Sequence):
    if len(loss_weights) != len(output_names):
      raise ValueError('`loss_weights` should have the same number of elements '
                       'as model output')
    return list(map(float, loss_weights))

  if isinstance(loss_weights, collections.Mapping):
    for name in output_names:
      if name not in loss_weights:
        raise ValueError('Loss weight for {} not found in `loss_weights` '
                         'dictionary.'.format(name))
    return [float(loss_weights[name]) for name in output_names]

  raise TypeError('`loss_weights` must be a list or a dict, '
                  'got {}'.format(str(loss_weights)))


def _prepare_metric_fns(metrics, output_names, loss_wrappers):
  """Converts `metrics` into a list of per-output list of metrics.

  Args:
    metrics: List of metrics to be evaluated during training and testing. Each
      metric can be specified using a string (e.g. `'accuracy'`) or a
      `tf.keras.metrics.Metric` object. For multi-output model, this can also be
      a dictionary like `{'output1': ['metric'], 'output2': ['metric2']}`.
      See the `metrics` argument in `tf.keras.Model.compile`.
    output_names: List of names of the model's output. If `metrics` is a
      dictionary, the names in this list will be taken as lookup keys.
    loss_wrappers: List of `_LossWrapper` objects corresponding to each output.

  Returns:
    A list of the same length as `output_names`, where each element is a list of
    callables representing the metrics to be evaluated on the corresponding
    output.
  """
  if metrics is None:
    return [[] for _ in output_names]

  if not isinstance(metrics, (list, collections.Mapping)):
    raise TypeError('`metrics` must be a list or a dict, got {}'.format(
        str(metrics)))

  to_list = lambda x: x if isinstance(x, list) else [x]

  if isinstance(metrics, collections.Mapping):
    # If `metrics` is a dictionary mapping output name to a list of metric fns,
    # coverts it to a list of lists using the order in `output_names`.
    metrics = [to_list(metrics.get(name, [])) for name in output_names]

  if not any(isinstance(m, list) for m in metrics):
    # If `metrics` is a list of metric fns, replicates them to be a list of
    # lists so that all metric fns can be applied to each output.
    metrics = [metrics for _ in output_names]

  # Here `metrics` is a list of lists, each sub-list corresponds to metric fns
  # to be applied on an output.
  if len(metrics) != len(output_names):
    raise ValueError('The number of sub-lists in `metrics` should be the '
                     'same as model output.')

  metric_fns = []
  for per_output_metrics, loss_wrapper in zip(metrics, loss_wrappers):
    metric_fns.append([
        keras.metrics.get(loss_wrapper.resolve_metric(metric))
        for metric in to_list(per_output_metrics)
    ])
  return metric_fns


def _compute_loss_and_metrics(losses,
                              metrics,
                              labels,
                              outputs,
                              sample_weights=None):
  """Computes total loss and (loss value, loss name) pairs for metrics.

  Args:
    losses: List of `_LossWrapper` objects to be evaluated on corresponding
      outputs. Must have the same length as `labels` and `outputs`.
    metrics: List of list of (metric fn, metric name) pairs, for additional
      metrics to report for each output. Must have the same length as `outputs`.
    labels: List of `Tensor` objects of ground truth targets. Must have the same
      length as `losses` and `outputs`.
    outputs: List of `Tensor` objects of predicted targets. Must have the same
      length as `losses` and `labels`.
    sample_weights: (optional) `Tensor` of weight for the loss of each sample.

  Returns:
    total_loss: Weighted sum of losses on all outputs.
    metrics: List of (value, name) pairs for metric reporting.
  """
  outputs = tf.nest.flatten(outputs)
  total_loss, output_metrics = [], []
  for (label, output, loss, per_output_metrics) in zip(labels, outputs, losses,
                                                       metrics):
    loss_value = loss(label, output, sample_weights)
    total_loss.append(loss.weight * loss_value)
    output_metrics.append((loss_value, loss.name))
    for metric_fn, metric_name in per_output_metrics:
      output_metrics.append((metric_fn(label, output), metric_name))
  return tf.add_n(total_loss), output_metrics


class AdversarialRegularization(keras.Model):
  """Wrapper thats adds adversarial regularization to a given `tf.keras.Model`.

  This model will reuse the layers and variables as the given `base_model`, so
  training this model will also update the variables in the `base_model`. The
  adversarial regularization can be configured by `adv_config`. (See
  `nsl.configs.AdvRegConfig` for the hyperparameters.) The regularization term
  will be added into training objective, and will be minimized during training
  together with other losses specified in `compile()`.

  This model expects its input to be a dictionary mapping feature names to
  feature values. The dictionary should contain both input data (`x`) and target
  data (`y`). The feature names of the target data should be passed to this
  model's constructor in `label_keys`, so the model can distinguish between
  input data and target data. If your samples are weighted, the sample weight
  should also be a feature in the dictionary, and its name should be passed to
  the constructor in `sample_weight_key`. When calling this model's `fit()` or
  `evaluate()` method, the argument `y` should not be set because the target
  data is already in the input dictionary. The dictionary format also implies
  that the input has to be named, i.e. the `name` argument of `tf.keras.Input()`
  should be set.

  Example:

  ```python
  # A linear regression model (for demonstrating the usage only)
  base_model = tf.keras.Sequential([
      tf.keras.Input(shape=(2,), name='input'),
      tf.keras.layers.Dense(1),
  ])

  # Applies the wrapper, with 0.2 as regularization weight.
  adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2)
  adv_model = nsl.keras.AdversarialRegularization(base_model,
                                                  label_keys=['label'],
                                                  adv_config=adv_config)

  # Compiles the model as usual.
  adv_model.compile(optimizer='adam', loss='mean_squared_error')

  # Trains the model. (The actual training data is omitted for clarity.)
  # The model minimizes (mean_squared_error + 0.2 * adversarial_regularization).
  adv_model.fit(x={'input': x_train, 'label': y_train}, batch_size=32)
  ```
  """

  def __init__(self,
               base_model,
               label_keys=('label',),
               sample_weight_key=None,
               adv_config=None):
    """Constructor of `AdversarialRegularization` class.

    Args:
      base_model: A `tf.Keras.Model` to which adversarial regularization will be
        applied.
      label_keys: A tuple of strings denoting which keys in the input features
        (a `dict` mapping keys to tensors) represent labels. This list should be
        1-to-1 corresponding to the output of the `base_model`.
      sample_weight_key: A string denoting which key in the input feature (a
        `dict` mapping keys to tensors) represents sample weight. If not set,
        the weight is 1.0 for each input example.
      adv_config: Instance of `nsl.configs.AdvRegConfig` for configuring
        adversarial regularization.
    """
    super(AdversarialRegularization,
          self).__init__(name='AdversarialRegularization')
    self.base_model = base_model
    self.label_keys = label_keys
    self.sample_weight_key = sample_weight_key
    self.adv_config = adv_config or nsl_configs.AdvRegConfig()

  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              **kwargs):
    if loss:
      self._compile_arg_loss = loss
      self._compile_arg_loss_weights = loss_weights
      self._compile_arg_metrics = metrics
      self._labeled_losses = None
      self._labeled_metrics = None

    # Compiles base model with saved losses and metrics.
    self.base_model.compile(
        optimizer,
        loss=self._compile_arg_loss,
        metrics=self._compile_arg_metrics,
        loss_weights=self._compile_arg_loss_weights,
        **kwargs)

    if hasattr(self.base_model, 'output_names'):
      # Organizes losses after the base model is fully compiled. The output
      # names from the base model is needed in case the loss (and/or
      # loss_weights) is specified in a dict().
      self._build_loss_and_metric_fns(self.base_model.output_names)

    # Hides losses and metrics for parent class so the model won't expect
    # separate label input (parameter `y`) in fit() and evaluate().
    super(AdversarialRegularization, self).compile(optimizer, **kwargs)

  def _make_metric_name(self, fn, label):
    """Generates a unique name, and resolves conflicts by appending a number."""
    if isinstance(fn, types.FunctionType):
      base_name = fn.__name__
    else:
      base_name = getattr(fn, 'name', fn.__class__.__name__)
    if len(self.label_keys) > 1:
      # If there are more than one output, disambigaute losses by corresponding
      # label name.
      base_name += '_' + label
    if base_name not in self._metric_name_count:
      self._metric_name_count[base_name] = 1
      return base_name
    else:
      self._metric_name_count[base_name] += 1
      return '{}_{}'.format(base_name, self._metric_name_count[base_name])

  def _build_loss_and_metric_fns(self, output_names):
    self._metric_name_count = collections.Counter()
    self._build_labeled_losses(output_names)
    self._build_labeled_metrics(output_names, self._labeled_losses)
    del self._metric_name_count  # no longer needed

  def _build_labeled_losses(self, output_names):
    if self._labeled_losses:
      return  # Losses are already populated.

    if len(output_names) != len(self.label_keys):
      raise ValueError('The model has different number of outputs and labels. '
                       '({} vs. {})'.format(
                           len(output_names), len(self.label_keys)))

    loss_fns = _prepare_loss_fns(self._compile_arg_loss, output_names)
    loss_weights = _prepare_loss_weights(self._compile_arg_loss_weights,
                                         output_names)
    self._labeled_losses = []
    for loss_fn, loss_weight, label_key in zip(loss_fns, loss_weights,
                                               self.label_keys):
      loss_name = self._make_metric_name(loss_fn, label_key)
      self._labeled_losses.append(_LossWrapper(loss_fn, loss_name, loss_weight))

  def _build_labeled_metrics(self, output_names, labeled_losses):
    if self._labeled_metrics:
      return  # Metrics are already populated.

    metric_fn_lists = _prepare_metric_fns(self._compile_arg_metrics,
                                          output_names, labeled_losses)
    self._labeled_metrics = []
    for metric_fns, label_key in zip(metric_fn_lists, self.label_keys):
      per_output_metrics = []
      for metric_fn in metric_fns:
        metric_name = self._make_metric_name(metric_fn, label_key)
        per_output_metrics.append((metric_fn, metric_name))
      self._labeled_metrics.append(per_output_metrics)

  def _get_or_create_base_output_names(self, outputs):
    num_output = len(tf.nest.flatten(outputs))
    return getattr(self.base_model, 'output_names',
                   ['output_%d' % i for i in range(1, num_output + 1)])

  def _compute_total_loss(self, labels, outputs, sample_weights=None):
    loss, _ = _compute_loss_and_metrics(self._labeled_losses,
                                        self._labeled_metrics, labels, outputs,
                                        sample_weights)
    return loss

  def _split_inputs(self, inputs):
    sample_weights = inputs.get(self.sample_weight_key, None)
    # Labels shouldn't be perturbed when generating adversarial examples.
    labels = [
        tf.stop_gradient(inputs[label_key]) for label_key in self.label_keys
    ]
    # Removes labels and sample weights from the input dictionary, since they
    # are only used in this class and base model does not need them as inputs.
    non_feature_keys = set(self.label_keys).union([self.sample_weight_key])
    inputs = {
        key: value
        for key, value in six.iteritems(inputs)
        if key not in non_feature_keys
    }
    return inputs, labels, sample_weights

  def _forward_pass(self, inputs, labels, sample_weights, base_model_kwargs):
    """Runs the usual forward pass to compute outputs, loss, and metrics."""
    with tf.GradientTape() as tape:
      tape.watch(list(inputs.values()))
      outputs = self.base_model(inputs, **base_model_kwargs)
      # If the base_model is a subclassed model, its output_names are not
      # available before its first call. If it is a dynamic subclassed model,
      # its output_names are not available even after its first call, so we
      # create names to match the number of outputs.
      self._build_loss_and_metric_fns(
          self._get_or_create_base_output_names(outputs))
      labeled_loss, metrics = _compute_loss_and_metrics(self._labeled_losses,
                                                        self._labeled_metrics,
                                                        labels, outputs,
                                                        sample_weights)
    return outputs, labeled_loss, metrics, tape

  def call(self, inputs, **kwargs):
    if any(key not in inputs for key in self.label_keys):
      # This is to prevent "no loss to optimize" error when the first call to
      # the model is without label input.
      raise ValueError('Labels are not in the input. For predicting examples '
                       'without labels, please use the base model instead.')

    inputs, labels, sample_weights = self._split_inputs(inputs)
    outputs, labeled_loss, metrics, tape = self._forward_pass(
        inputs, labels, sample_weights, kwargs)
    self.add_loss(labeled_loss)
    for value, name in metrics:
      self.add_metric(value, aggregation='mean', name=name)

    # Adversarial loss.
    base_model_fn = lambda inputs: self.base_model(inputs, **kwargs)
    adv_loss = adversarial_loss(
        inputs,
        labels,
        base_model_fn,
        self._compute_total_loss,
        sample_weights=sample_weights,
        adv_config=self.adv_config,
        labeled_loss=labeled_loss,
        gradient_tape=tape)
    self.add_loss(self.adv_config.multiplier * adv_loss)
    self.add_metric(adv_loss, name='adversarial_loss', aggregation='mean')
    return outputs

  def perturb_on_batch(self, x, **config_kwargs):
    """Perturbs the given input to generates adversarial examples.

    Args:
      x: Input examples to be perturbed, in a dictionary of Numpy arrays,
        `Tensor`, `SparseTensor`, or `RaggedTensor` objects. The first
        dimension of all tensors or arrays should be the same (i.e. batch size).
      **config_kwargs: (optional) hyperparameters for generating adversarial
        preturbation. Any keyword argument here will overwrite the corresponding
        field in `nsl.configs.AdvNeighborConfig` specified in `__init__`.
        Acceptable keys: `feature_mask`, `adv_step_size`, and `adv_grad_norm`.

    Returns:
      A dictionary of NumPy arrays, `SparseTensor`, or `RaggedTensor` objects of
      the generated adversarial examples.
    """
    x = tf.nest.map_structure(tf.convert_to_tensor, x, expand_composites=True)
    inputs, labels, sample_weights = self._split_inputs(x)
    _, labeled_loss, _, tape = self._forward_pass(inputs, labels,
                                                  sample_weights,
                                                  {'training': False})

    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
    config = attr.evolve(self.adv_config.adv_neighbor_config, **config_kwargs)
    adv_inputs, _ = nsl_lib.gen_adv_neighbor(
        inputs, labeled_loss, config=config, gradient_tape=tape)

    if tf.executing_eagerly():
      # Converts `Tensor` objects to NumPy arrays and keeps other objects (e.g.
      # `SparseTensor`) as-is.
      adv_inputs = tf.nest.map_structure(
          lambda x: x.numpy() if hasattr(x, 'numpy') else x,
          adv_inputs,
          expand_composites=False)
    else:
      adv_inputs = keras.backend.function([], adv_inputs)([])

    # Inserts the labels and sample_weights back to the input dictionary, so
    # the returned input has the same structure as the original input.
    for label_key, label in zip(self.label_keys, labels):
      adv_inputs[label_key] = label
    if self.sample_weight_key is not None:
      adv_inputs[self.sample_weight_key] = sample_weights

    return adv_inputs
