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
"""Tests for neural_structured_learning.keras.adversarial_regularization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import neural_structured_learning.configs as configs
from neural_structured_learning.keras import adversarial_regularization
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


def build_linear_keras_sequential_model(input_shape, weights):
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=input_shape, name='feature'))
  model.add(
      tf.keras.layers.Dense(
          weights.shape[-1],
          use_bias=False,
          kernel_initializer=tf.keras.initializers.Constant(weights)))
  return model


def build_linear_keras_functional_model(input_shape,
                                        weights,
                                        input_name='feature'):
  inputs = tf.keras.Input(shape=input_shape, name=input_name)
  layer = tf.keras.layers.Dense(
      weights.shape[-1],
      use_bias=False,
      kernel_initializer=tf.keras.initializers.Constant(weights))
  outputs = layer(inputs)
  return tf.keras.Model(inputs={input_name: inputs}, outputs=outputs)


def build_linear_keras_subclassed_model(input_shape, weights, dynamic=False):
  del input_shape

  class LinearModel(tf.keras.Model):

    def __init__(self):
      super(LinearModel, self).__init__(dynamic=dynamic)
      self.dense = tf.keras.layers.Dense(
          weights.shape[-1],
          use_bias=False,
          name='dense',
          kernel_initializer=tf.keras.initializers.Constant(weights))

    def call(self, inputs):
      return self.dense(inputs['feature'])

  return LinearModel()


def build_linear_keras_dynamic_model(input_shape, weights):
  return build_linear_keras_subclassed_model(input_shape, weights, dynamic=True)


class AdversarialLossTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(AdversarialLossTest, self).setUp()
    self.adv_step_size = 0.01
    self.adv_config = configs.make_adv_reg_config(
        adv_step_size=self.adv_step_size, adv_grad_norm='infinity')

  def _build_linear_regression_model_and_inputs(self, model_fn):
    # The model and input are shared across test cases.
    w = np.array([[4.0], [-3.0]])
    x0 = np.array([[2.0, 3.0]])
    y0 = np.array([[0.0]])
    y_hat = np.dot(x0, w)
    x_adv = x0 + self.adv_step_size * np.sign((y_hat - y0) * w.T)
    y_hat_adv = np.dot(x_adv, w)
    model = model_fn(input_shape=(2,), weights=w)
    loss_fn = tf.keras.losses.MeanSquaredError()
    inputs = {'feature': tf.constant(x0)}
    labels = tf.constant(y0)
    expected_adv_loss = np.reshape((y_hat_adv - y0)**2, ())

    # Initializes the variables in TF 1.x. This is a no-op in TF 2.0.
    self.evaluate(tf.compat.v1.global_variables_initializer())
    return inputs, labels, model, loss_fn, expected_adv_loss

  def evaluate(self, *args, **kwargs):
    if hasattr(tf.keras.backend, 'get_session'):
      # Sets the Keras Session as default TF Session, so that the variable
      # in Keras subclassed model can be initialized correctly. The variable
      # is not created until the first call to the model, so the initialization
      # is not captured in the global_variables_initializer above.
      with tf.keras.backend.get_session().as_default():
        return super(AdversarialLossTest, self).evaluate(
            *args, **kwargs)
    else:
      return super(AdversarialLossTest, self).evaluate(
          *args, **kwargs)

  @parameterized.named_parameters([
      ('sequential', build_linear_keras_sequential_model),
      ('functional', build_linear_keras_functional_model),
      ('subclassed', build_linear_keras_subclassed_model),
  ])
  def test_normal_case(self, model_fn):
    inputs, labels, model, loss_fn, expected_adv_loss = (
        self._build_linear_regression_model_and_inputs(model_fn))
    adv_loss = adversarial_regularization.adversarial_loss(
        inputs, labels, model, loss_fn, adv_config=self.adv_config)
    self.assertAllClose(expected_adv_loss, self.evaluate(adv_loss))

  @parameterized.named_parameters([
      ('sequential', build_linear_keras_sequential_model),
      ('functional', build_linear_keras_functional_model),
      ('subclassed', build_linear_keras_subclassed_model),
  ])
  def test_given_predictions(self, model_fn):
    inputs, labels, model, loss_fn, expected_adv_loss = (
        self._build_linear_regression_model_and_inputs(model_fn))

    with tf.GradientTape() as tape:
      tape.watch(inputs['feature'])
      outputs = model(inputs)

    # Wraps self.model to record the number of times it gets called. The counter
    # cannot be a local variable because assignments to names always go into the
    # innermost scope.
    # https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces
    call_count = collections.Counter()

    def wrapped_model(inputs):
      call_count['model'] += 1
      return model(inputs)

    adv_loss = adversarial_regularization.adversarial_loss(
        inputs,
        labels,
        wrapped_model,
        loss_fn,
        adv_config=self.adv_config,
        predictions=outputs,
        gradient_tape=tape)
    self.assertAllClose(expected_adv_loss, self.evaluate(adv_loss))
    # The model should be called only once, i.e. not re-calculating the
    # predictions on original inputs.
    self.assertEqual(1, call_count['model'])

  @parameterized.named_parameters([
      ('sequential', build_linear_keras_sequential_model),
      ('functional', build_linear_keras_functional_model),
      ('subclassed', build_linear_keras_subclassed_model),
  ])
  def test_given_labeled_loss(self, model_fn):
    inputs, labels, model, loss_fn, expected_adv_loss = (
        self._build_linear_regression_model_and_inputs(model_fn))

    with tf.GradientTape() as tape:
      tape.watch(inputs['feature'])
      outputs = model(inputs)
      labeled_loss = loss_fn(labels, outputs)

    # Wraps self.model and self.loss_fn to record the number of times they get
    # called.
    call_count = collections.Counter()

    def wrapped_model(inputs):
      call_count['model'] += 1
      return model(inputs)

    def wrapped_loss_fn(*args, **kwargs):
      call_count['loss_fn'] += 1
      return loss_fn(*args, **kwargs)

    adv_loss = adversarial_regularization.adversarial_loss(
        inputs,
        labels,
        wrapped_model,
        wrapped_loss_fn,
        adv_config=self.adv_config,
        labeled_loss=labeled_loss,
        gradient_tape=tape)
    self.assertAllClose(expected_adv_loss, self.evaluate(adv_loss))
    # The model and loss_fn should be called only once, i.e. not re-calculating
    # the predictions and/or loss on original inputs.
    self.assertEqual(1, call_count['model'])
    self.assertEqual(1, call_count['loss_fn'])

  def test_with_model_kwargs(self):
    w = np.array([[4.0], [-3.0]])
    x0 = np.array([[2.0, 3.0]])
    y0 = np.array([[0.0]])
    model = build_linear_keras_sequential_model(input_shape=(2,), weights=w)
    model.add(tf.keras.layers.BatchNormalization())

    adv_loss = adversarial_regularization.adversarial_loss(
        features={'feature': tf.constant(x0)},
        labels=tf.constant(y0),
        model=model,
        loss_fn=tf.keras.losses.MeanSquaredError(),
        adv_config=self.adv_config,
        model_kwargs={'training': True})
    # BatchNormalization returns 0 for signle-example batch when training=True.
    self.assertAllClose(0.0, self.evaluate(adv_loss))


class AdversarialRegularizationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ('sequential', build_linear_keras_sequential_model),
      ('functional', build_linear_keras_functional_model),
      ('subclassed', build_linear_keras_subclassed_model),
  ])
  def test_predict_by_adv_model_with_labels(self, model_fn):
    model = model_fn(input_shape=(2,), weights=np.array([[1.0], [-1.0]]))
    inputs = {
        'feature': tf.constant([[5.0, 3.0]]),
        'label': tf.constant([[1.0]])
    }

    adv_model = adversarial_regularization.AdversarialRegularization(model)
    adv_model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss='MSE')

    prediction = adv_model.predict(x=inputs, steps=1, batch_size=1)
    self.assertAllEqual([[2.0]], prediction)

  @parameterized.named_parameters([
      ('sequential', build_linear_keras_sequential_model),
      ('functional', build_linear_keras_functional_model),
      ('subclassed', build_linear_keras_subclassed_model),
  ])
  def test_predict_by_base_model(self, model_fn):
    model = model_fn(input_shape=(2,), weights=np.array([[1.0], [-1.0]]))
    inputs = {'feature': tf.constant([[5.0, 3.0]])}

    adv_model = adversarial_regularization.AdversarialRegularization(model)
    adv_model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss='MSE')

    prediction = model.predict(x=inputs, steps=1, batch_size=1)
    self.assertAllEqual([[2.0]], prediction)

  def _set_up_linear_regression(self, sample_weight=1.0):
    w = np.array([[4.0], [-3.0]])
    x0 = np.array([[2.0, 3.0]])
    y0 = np.array([[0.0]])
    adv_multiplier = 0.2
    adv_step_size = 0.01
    learning_rate = 0.01
    adv_config = configs.make_adv_reg_config(
        multiplier=adv_multiplier,
        adv_step_size=adv_step_size,
        adv_grad_norm='infinity')
    y_hat = np.dot(x0, w)
    x_adv = x0 + adv_step_size * np.sign((y_hat - y0) * w.T)
    y_hat_adv = np.dot(x_adv, w)
    grad_w_labeled_loss = sample_weight * 2. * (y_hat - y0) * x0.T
    grad_w_adv_loss = adv_multiplier * sample_weight * 2. * (y_hat_adv -
                                                             y0) * x_adv.T
    w_new = w - learning_rate * (grad_w_labeled_loss + grad_w_adv_loss)
    return w, x0, y0, learning_rate, adv_config, w_new

  @parameterized.named_parameters([
      ('sequential', build_linear_keras_sequential_model),
      ('functional', build_linear_keras_functional_model),
      ('subclassed', build_linear_keras_subclassed_model),
  ])
  def test_train_fgsm(self, model_fn):
    w, x0, y0, lr, adv_config, w_new = self._set_up_linear_regression()

    inputs = {'feature': tf.constant(x0), 'label': tf.constant(y0)}
    model = model_fn(input_shape=(2,), weights=w)
    adv_model = adversarial_regularization.AdversarialRegularization(
        model, label_keys=['label'], adv_config=adv_config)
    adv_model.compile(optimizer=tf.keras.optimizers.SGD(lr), loss='MSE')
    adv_model.fit(x=inputs, batch_size=1, steps_per_epoch=1)

    self.assertAllClose(w_new, tf.keras.backend.get_value(model.weights[0]))

  def test_train_fgsm_functional_model_diff_feature_key(self):
    # This test asserts that AdversarialRegularization works regardless of the
    # alphabetical order of feature and label keys in the input dictionary. This
    # is specifically for Keras Functional models because those models sort the
    # inputs by key.
    w, x0, y0, lr, adv_config, w_new = self._set_up_linear_regression()

    inputs = {'the_feature': tf.constant(x0), 'label': tf.constant(y0)}
    model = build_linear_keras_functional_model(
        input_shape=(2,), weights=w, input_name='the_feature')
    adv_model = adversarial_regularization.AdversarialRegularization(
        model, label_keys=['label'], adv_config=adv_config)
    adv_model.compile(optimizer=tf.keras.optimizers.SGD(lr), loss='MSE')
    adv_model.fit(x=inputs, batch_size=1, steps_per_epoch=1)

    self.assertAllClose(w_new, tf.keras.backend.get_value(model.weights[0]))

  @parameterized.named_parameters([
      ('sequential', build_linear_keras_sequential_model),
      ('functional', build_linear_keras_functional_model),
      ('subclassed', build_linear_keras_subclassed_model),
  ])
  def test_train_fgsm_with_sample_weights(self, model_fn):
    sample_weight = np.array([[2.0]])
    w, x0, y0, lr, adv_config, w_new = self._set_up_linear_regression(
        sample_weight)

    inputs = {
        'feature': tf.constant(x0),
        'label': tf.constant(y0),
        'sample_weight': tf.constant(sample_weight)
    }
    model = model_fn(input_shape=(2,), weights=w)
    adv_model = adversarial_regularization.AdversarialRegularization(
        model,
        label_keys=['label'],
        sample_weight_key='sample_weight',
        adv_config=adv_config)
    adv_model.compile(optimizer=tf.keras.optimizers.SGD(lr), loss='MSE')
    adv_model.fit(x=inputs, batch_size=1, steps_per_epoch=1)

    self.assertAllClose(w_new, tf.keras.backend.get_value(model.weights[0]))

  @parameterized.named_parameters([
      ('sequential', build_linear_keras_sequential_model),
      ('functional', build_linear_keras_functional_model),
      ('subclassed', build_linear_keras_subclassed_model),
      ('dynamic', build_linear_keras_dynamic_model),
  ])
  @test_util.run_v2_only
  def test_train_with_distribution_strategy(self, model_fn):
    w, x0, y0, lr, adv_config, w_new = self._set_up_linear_regression()
    inputs = tf.data.Dataset.from_tensor_slices({
        'feature': x0,
        'label': y0
    }).batch(1)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      model = model_fn(input_shape=(2,), weights=w)
      adv_model = adversarial_regularization.AdversarialRegularization(
          model, label_keys=['label'], adv_config=adv_config)
      adv_model.compile(
          optimizer=tf.keras.optimizers.SGD(lr), loss='MSE', metrics=['mae'])

    adv_model.fit(x=inputs)

    self.assertAllClose(w_new, tf.keras.backend.get_value(model.weights[0]))

  def test_train_with_loss_object(self):
    w, x0, y0, lr, adv_config, w_new = self._set_up_linear_regression()

    inputs = {'feature': tf.constant(x0), 'label': tf.constant(y0)}
    model = build_linear_keras_functional_model(input_shape=(2,), weights=w)
    adv_model = adversarial_regularization.AdversarialRegularization(
        model, label_keys=['label'], adv_config=adv_config)
    adv_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr),
        loss=tf.keras.losses.MeanSquaredError())
    adv_model.fit(x=inputs, batch_size=1, steps_per_epoch=1)

    self.assertAllClose(w_new, tf.keras.backend.get_value(model.weights[0]))

  def test_train_with_metrics(self):
    w, x0, y0, lr, adv_config, _ = self._set_up_linear_regression()

    inputs = {'feature': tf.constant(x0), 'label': tf.constant(y0)}
    model = build_linear_keras_functional_model(input_shape=(2,), weights=w)
    adv_model = adversarial_regularization.AdversarialRegularization(
        model, label_keys=['label'], adv_config=adv_config)
    adv_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr), loss='MSE', metrics=['mae'])
    history = adv_model.fit(x=inputs, batch_size=1, steps_per_epoch=1)

    actual_labeled_loss = history.history['mean_squared_error'][0]
    actual_metric = history.history['mean_absolute_error'][0]
    expected_labeled_loss = np.power(y0 - np.dot(x0, w), 2).mean()
    expected_metric = np.abs(y0 - np.dot(x0, w)).mean()
    self.assertAllClose(expected_labeled_loss, actual_labeled_loss)
    self.assertAllClose(expected_metric, actual_metric)

  def test_train_with_duplicated_metrics(self):
    w, x0, y0, lr, adv_config, _ = self._set_up_linear_regression()

    inputs = {'feature': tf.constant(x0), 'label': tf.constant(y0)}
    model = build_linear_keras_functional_model(input_shape=(2,), weights=w)
    adv_model = adversarial_regularization.AdversarialRegularization(
        model, label_keys=['label'], adv_config=adv_config)
    adv_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr), loss=['MSE'], metrics=[['MSE']])
    history = adv_model.fit(x=inputs, batch_size=1, steps_per_epoch=1)

    self.assertIn('mean_squared_error', history.history)
    self.assertIn('mean_squared_error_2', history.history)
    self.assertEqual(history.history['mean_squared_error'],
                     history.history['mean_squared_error_2'])

  def test_train_with_metric_object(self):
    w, x0, y0, lr, adv_config, _ = self._set_up_linear_regression()

    inputs = {'feature': tf.constant(x0), 'label': tf.constant(y0)}
    model = build_linear_keras_functional_model(input_shape=(2,), weights=w)
    adv_model = adversarial_regularization.AdversarialRegularization(
        model, label_keys=['label'], adv_config=adv_config)
    adv_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr),
        loss='MSE',
        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    history = adv_model.fit(x=inputs, batch_size=1, steps_per_epoch=1)

    actual_metric = history.history['mean_absolute_error'][0]
    expected_metric = np.abs(y0 - np.dot(x0, w)).mean()
    self.assertAllClose(expected_metric, actual_metric)

  def test_train_with_2_outputs(self):
    w, x0, y0, lr, adv_config, _ = self._set_up_linear_regression()
    inputs = {
        'feature': tf.constant(x0),
        'label1': tf.constant(y0),
        'label2': tf.constant(-y0)
    }

    input_layer = tf.keras.Input(shape=(2,), name='feature')
    layer1 = tf.keras.layers.Dense(
        w.shape[-1],
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(w))
    layer2 = tf.keras.layers.Dense(
        w.shape[-1],
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(-w))
    model = tf.keras.Model(
        inputs={'feature': input_layer},
        outputs=[layer1(input_layer), layer2(input_layer)])

    adv_model = adversarial_regularization.AdversarialRegularization(
        model, label_keys=['label1', 'label2'], adv_config=adv_config)
    adv_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr),
        loss='MSE',
        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    history = adv_model.fit(x=inputs, batch_size=1, steps_per_epoch=1)

    expected_metric = np.abs(y0 - np.dot(x0, w)).mean()
    self.assertAllClose(expected_metric,
                        history.history['mean_absolute_error_label1'][0])
    self.assertAllClose(expected_metric,
                        history.history['mean_absolute_error_label2'][0])

  def test_evaluate_binary_classification_metrics(self):
    # multi-label binary classification model
    w = np.array([[4.0, 1.0, -5.0], [-3.0, 1.0, 2.0]])
    x0 = np.array([[2.0, 3.0]])
    y0 = np.array([[0.0, 1.0, 1.0]])
    inputs = {'feature': tf.constant(x0), 'label': tf.constant(y0)}
    model = build_linear_keras_sequential_model(input_shape=(2,), weights=w)
    model.add(tf.keras.layers.Lambda(tf.sigmoid))

    adv_model = adversarial_regularization.AdversarialRegularization(model)
    adv_model.compile(
        optimizer=tf.keras.optimizers.SGD(0.1),
        loss='squared_hinge',
        metrics=['accuracy', 'ce'])
    metrics_values = adv_model.evaluate(inputs, steps=1)
    results = dict(zip(adv_model.metrics_names, metrics_values))

    y_hat = 1. / (1. + np.exp(-np.dot(x0, w)))  # [[0.26894, 0.99331, 0.01799]]
    accuracy = np.mean(np.sign(y_hat - 0.5) == np.sign(y0 - 0.5))  # (1+1+0) / 3
    cross_entropy = np.mean(y0 * -np.log(y_hat) + (1 - y0) * -np.log(1 - y_hat))

    self.assertIn('binary_accuracy', results)
    self.assertIn('binary_crossentropy', results)
    self.assertAllClose(accuracy, results['binary_accuracy'])
    self.assertAllClose(cross_entropy, results['binary_crossentropy'])

  def test_evaluate_classification_metrics(self):
    # multi-class logistic regression model
    w = np.array([[4.0, 1.0, -5.0], [-3.0, 1.0, 2.0]])
    x0 = np.array([[2.0, 3.0]])
    y0 = np.array([[1]])
    inputs = {'feature': tf.constant(x0), 'label': tf.constant(y0)}
    model = build_linear_keras_sequential_model(input_shape=(2,), weights=w)
    model.add(tf.keras.layers.Softmax())

    adv_model = adversarial_regularization.AdversarialRegularization(model)
    adv_model.compile(
        optimizer=tf.keras.optimizers.SGD(0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'ce'])
    metrics_values = adv_model.evaluate(inputs, steps=1)
    results = dict(zip(adv_model.metrics_names, metrics_values))

    logit = np.dot(x0, w)  # [[-1.,  5., -4.]]
    accuracy = np.mean(np.argmax(logit, axis=-1) == y0)
    cross_entropy = np.log(np.sum(np.exp(logit))) - np.reshape(logit[:, y0], ())

    self.assertIn('sparse_categorical_accuracy', results)
    self.assertIn('sparse_categorical_crossentropy', results)
    self.assertAllClose(accuracy, results['sparse_categorical_accuracy'])
    self.assertAllClose(cross_entropy,
                        results['sparse_categorical_crossentropy'])

  def test_perturb_on_batch(self):
    w, x0, y0, lr, adv_config, _ = self._set_up_linear_regression()
    inputs = {'feature': x0, 'label': y0}
    model = build_linear_keras_functional_model(input_shape=(2,), weights=w)
    adv_model = adversarial_regularization.AdversarialRegularization(
        model, label_keys=['label'], adv_config=adv_config)
    adv_model.compile(optimizer=tf.keras.optimizers.SGD(lr), loss=['MSE'])
    adv_inputs = adv_model.perturb_on_batch(inputs)

    y_hat = np.dot(x0, w)
    adv_step_size = adv_config.adv_neighbor_config.adv_step_size
    x_adv = x0 + adv_step_size * np.sign((y_hat - y0) * w.T)
    self.assertAllClose(x_adv, adv_inputs['feature'])
    self.assertAllClose(y0, adv_inputs['label'])

  def test_perturb_on_batch_custom_config(self):
    w, x0, y0, lr, adv_config, _ = self._set_up_linear_regression()
    inputs = {'feature': x0, 'label': y0}
    model = build_linear_keras_functional_model(input_shape=(2,), weights=w)
    adv_model = adversarial_regularization.AdversarialRegularization(
        model, label_keys=['label'], adv_config=adv_config)
    adv_model.compile(optimizer=tf.keras.optimizers.SGD(lr), loss=['MSE'])

    adv_step_size = 0.2  # A different value from config.adv_step_size
    adv_inputs = adv_model.perturb_on_batch(inputs, adv_step_size=adv_step_size)

    y_hat = np.dot(x0, w)
    x_adv = x0 + adv_step_size * np.sign((y_hat - y0) * w.T)
    self.assertAllClose(x_adv, adv_inputs['feature'])
    self.assertAllClose(y0, adv_inputs['label'])


if __name__ == '__main__':
  tf.test.main()
