# Copyright 2022 Google LLC
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

r"""Finetuning a linear probe on top of the model.

This code finetunes a linear probe on top of a given model. This is to be used
as a baseline for the provided method.
"""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from meo.mlp_baseline import configs
from meo.mlp_baseline import data_utils
from meo.mlp_baseline import extended_model
from meo.mlp_baseline import obfuscations
import numpy as np
import tensorflow as tf


_EPOCHS = flags.DEFINE_integer('epochs', 50, 'Number of training epochs.')

_TPU = flags.DEFINE_string('tpu', None, 'TPU to use.')
_BASE_LR = flags.DEFINE_float('base_lr', 1e-4, 'Initial learning rate.')
_LR_DECAY_TYPE = flags.DEFINE_enum(
    'lr_decay_type', 'exponential', ['cosine', 'exponential'],
    'Type of learning rate decay.')
_LR_DECAY_FACTOR = flags.DEFINE_float(
    'lr_decay_factor', 0.5, 'Factor with which to decay learning rate per decay'
    'epoch.')
_LR_DECAY_EPOCHS = flags.DEFINE_integer(
    'lr_decay_epochs', 10, 'Number of epochs per learning rate decay step.')
_FINETUNE = flags.DEFINE_bool(
    'finetune', False, 'Choose whether to finetune the base model.')

_MOMENTUM = flags.DEFINE_float('momentum', 0.9, 'Momentum.')
_WEIGHT_DECAY = flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay.')


_DATASET = flags.DEFINE_enum(
    'dataset', 'imagenet', ['imagenet', 'coco'],
    'Dataset to be used for the experiment.')

_INPUT_FEATURE_NAME = flags.DEFINE_enum(
    'input_feature_name', 'embed', ['pixel', 'embed'],
    'Format of input data (choice between pixel i.e. raw images and embed i.e.'
    'representations produced by target model'
)

_OBFUSCATIONS_TRAIN = flags.DEFINE_list(
    'obfuscations_train', data_utils.TRAIN_OBFUSCATIONS,
    'List of obfuscations to apply to the training data.')
_OBFUSCATIONS_EVAL = flags.DEFINE_list(
    'obfuscations_eval', data_utils.EVAL_OBFUSCATIONS,
    'List of obfuscations to apply to the evaluation data.')

_NUM_CORES = flags.DEFINE_integer('num_cores', None, 'Number of TPU cores.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, 'Batch size (per core).')
_DATA_DIR_TRAIN = flags.DEFINE_string(
    'data_dir_train', None, 'Path to training data.')
_DATA_DIR_EVAL = flags.DEFINE_string(
    'data_dir_eval', None, 'Path to eval data.')
_MODEL_DIR = flags.DEFINE_string('model_dir', None, 'Path to model.')

_MODEL_TYPE = flags.DEFINE_enum(
    'model_type', 'align',
    ['resnet50', 'resnet101', 'resnet152', 'bit-m152x4', 'align', 'vitb32'],
    'Base model extractor. See configs.ModelConfig for a list of available'
    'models.')

_NUM_SAVE_IMG = flags.DEFINE_integer(
    'num_save_img', 1, 'Number images to save in summary.')
_LOGGING_STEP = flags.DEFINE_integer(
    'logging_step', 20, 'Interval of steps between log messages.')

FLAGS = flags.FLAGS


def train_and_eval_model(
    model_clf: extended_model.FeatureExtractorWithClassifier,
    train_dataset: tf.distribute.DistributedDataset,
    train_steps_per_epoch: int,
    test_dataset: tf.distribute.DistributedDataset,
    eval_steps_per_epoch: int,
    optimizer: tf.keras.optimizers.Optimizer,
    strategy: tf.distribute.Strategy,
) -> None:
  """Train and evaluate the embeddings and the classifier on top them.

  Args:
    model_clf: Model to train.
    train_dataset: Dataset over which to perform training. This dataset is
      comprised of tf.train.Example records, containing features 'image' and
      'label', corresponding to the obfuscated image and its respective label.
    train_steps_per_epoch: Number of training steps per epoch.
    test_dataset: Dataset over which to evaluate the model.
    eval_steps_per_epoch: Number of evaluation steps per epoch.
    optimizer: Optimizer to use.
    strategy: Strategy to be used for distributed training.
  """
  with strategy.scope():
    loss_names = [
        'clf_train_loss', 'clf_train_classification_loss', 'clf_train_reg_loss',
        'clf_train_embed_loss'
    ]
    for obf in _OBFUSCATIONS_EVAL.value:
      loss_names.append('clf_test_loss_{}'.format(obf))
    loss_metrics = {
        loss_name: tf.keras.metrics.Mean(loss_name, dtype=tf.float32)
        for loss_name in loss_names
    }

    acc_names = ['clf_test_acc_worst']
    for obf in _OBFUSCATIONS_EVAL.value:
      acc_names.append('clf_test_acc_{}'.format(obf))
    acc_metrics = {
        acc_name:
        tf.keras.metrics.SparseCategoricalAccuracy(acc_name, dtype=tf.float32)
        for acc_name in acc_names
    }

    clf_checkpoint = tf.train.Checkpoint(model=model_clf, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(_MODEL_DIR.value)
    initial_epoch = 0
    if latest_checkpoint:
      clf_checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // train_steps_per_epoch

  clf_train_summary_writer = tf.summary.create_file_writer(
      os.path.join(_MODEL_DIR.value, 'summaries/clf_train'))

  clf_test_summary_writer = tf.summary.create_file_writer(
      os.path.join(_MODEL_DIR.value, 'summaries/clf_test'))

  @tf.function
  def clf_train_step(item):

    def step_fn(inputs):
      if _INPUT_FEATURE_NAME.value == 'pixel':
        all_data = tf.stack([
            inputs['image_{}'.format(obf)]
            for obf in _OBFUSCATIONS_TRAIN.value[1:]
        ], axis=1)
        labels = inputs['label']

        if all_data.dtype == tf.uint8:
          all_data = tf.cast(all_data, tf.float32) / 255.0

      elif _INPUT_FEATURE_NAME.value == 'embed':
        all_data = inputs[0]
        labels = inputs[1]

      with tf.GradientTape() as tape:
        if _INPUT_FEATURE_NAME.value == 'pixel':
          # Pack the images from the shape of [batch_size, num_views, H, W, C]
          # to [batch_size * num_views, H, W, C] to compute embeddings from the
          # encoder model.
          orig_shape = tf.shape(all_data)
          num_views = orig_shape[1]
          height = orig_shape[2]
          width = orig_shape[3]
          channels = orig_shape[4]
          all_data = tf.reshape(
              all_data, [-1, height, width, channels]
          )
          all_embed = model_clf.feature_extractor.encode_clean(
              all_data, training=False)
        else:
          all_embed = all_data
          orig_shape = tf.shape(all_data)
          num_views = orig_shape[1]
          all_embed = tf.reshape(
              all_embed,
              [-1, tf.shape(all_embed)[-1]]
          )

        # Crossentropy loss for the classifier.
        logits = model_clf.clf_layer(all_embed, training=True)

        # Duplicate the labels by the number of views.
        labels = tf.tile(tf.expand_dims(labels, axis=-1), [1, num_views])
        labels = tf.reshape(labels, [-1])

        crossentropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits
        )

        # model_clf.losses contains the regularizer for the linear classifier.
        regularizer_loss = tf.reduce_sum(model_clf.losses)

        loss = crossentropy_loss + regularizer_loss

        # Divide by number of replicas to balance out the artificially increased
        # batch size.
        per_replica_loss = loss / strategy.num_replicas_in_sync

      gradients = tape.gradient(per_replica_loss, model_clf.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model_clf.trainable_variables))
      loss_metrics['clf_train_loss'].update_state(loss)
      loss_metrics['clf_train_classification_loss'].update_state(
          crossentropy_loss)
      loss_metrics['clf_train_reg_loss'].update_state(regularizer_loss)

    strategy.run(step_fn, args=(item,))

  @tf.function
  def clf_test_step(item):

    def step_fn(inputs):
      all_logits = []
      labels = inputs['label']
      for obf in _OBFUSCATIONS_EVAL.value:
        images = inputs['image_{}'.format(obf)]
        if images.dtype == tf.uint8:
          images = tf.cast(images, tf.float32) / 255.0

        # Recover embeddings from the base model and then classify them,
        # skipping the autoencoder part for inference.
        embeds = model_clf.feature_extractor.base_model(images)
        logits = model_clf.clf_layer(embeds, training=False)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        loss = tf.math.divide_no_nan(
            tf.reduce_sum(losses), tf.cast(tf.size(losses), dtype=tf.float32))

        loss_metrics['clf_test_loss_{}'.format(obf)].update_state(loss)
        acc_metrics['clf_test_acc_{}'.format(obf)].update_state(
            labels, tf.nn.softmax(logits))

        if obf in data_utils.HOLD_OUT_OBFUSCATIONS:
          all_logits.append(tf.expand_dims(logits, 1))

      all_logits = tf.concat(all_logits, axis=1)
      all_labels = tf.tile(tf.expand_dims(labels, 1),
                           [1, len(data_utils.HOLD_OUT_OBFUSCATIONS)])
      error = tf.nn.sparse_softmax_cross_entropy_with_logits(
          all_labels, all_logits)
      indices = tf.argmax(error, axis=1)
      worst_logits = tf.gather(all_logits, indices, batch_dims=1)
      acc_metrics['clf_test_acc_worst'].update_state(labels, worst_logits)

    strategy.run(step_fn, args=(item,))

  train_iterator = iter(train_dataset)
  for epoch in range(initial_epoch, _EPOCHS.value):
    logging.info('Training Epoch: %s', epoch)
    with clf_train_summary_writer.as_default():
      for step in range(train_steps_per_epoch):
        if step % _LOGGING_STEP.value == 0:
          logging.info('Training Epoch %s, Step %s', epoch, step)
        item = next(train_iterator)
        clf_train_step(item)

      logging.info(
          'Train Loss: %s',
          round(loss_metrics['clf_train_loss'].result().numpy(), 3)
      )

      for loss_name, loss in loss_metrics.items():
        if 'train' in loss_name:
          tf.summary.scalar(loss_name, loss.result().numpy(),
                            step=optimizer.iterations)
          loss.reset_state()

      if _LR_DECAY_TYPE.value == 'exponential':
        lr_sched = tf.keras.optimizers.schedules.ExponentialDecay.from_config(
            optimizer.get_config()['learning_rate']['config']
        )
      elif _LR_DECAY_TYPE.value == 'cosine':
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts.from_config(
            optimizer.get_config()['learning_rate']['config']
        )

      tf.summary.scalar(
          'clf_learning_rate',
          lr_sched(optimizer.iterations).numpy(),
          step=optimizer.iterations)

    logging.info('Testing Epoch: %s', epoch)
    with clf_test_summary_writer.as_default():
      test_iterator = iter(test_dataset)
      for step in range(eval_steps_per_epoch):
        item = next(test_iterator)
        if step % _LOGGING_STEP.value == 0:
          logging.info('Testing Epoch %s, Step %s', epoch, step)
        clf_test_step(item)

      for loss_name, loss in loss_metrics.items():
        if 'test' in loss_name:
          tf.summary.scalar(loss_name, loss.result().numpy(),
                            step=optimizer.iterations)
          loss.reset_state()

      for acc_name, acc in acc_metrics.items():
        if 'test' in acc_name:
          tf.summary.scalar(acc_name, acc.result().numpy(),
                            step=optimizer.iterations)
          acc.reset_state()

    clf_checkpoint.save(os.path.join(_MODEL_DIR.value, 'clf_checkpoint'))


def main(argv: Sequence[str]) -> None:
  del argv
  tf.config.set_soft_device_placement(True)

  # Ensure that the first obfuscation in the training obfuscations is 'Clean'.
  if _OBFUSCATIONS_TRAIN.value[0] != data_utils.CLEAN:
    raise ValueError('The first of the training obfuscations must be the (noop)'
                     'obfuscation \'{}\''.format(data_utils.CLEAN))

  per_replica_batch_size = _BATCH_SIZE.value
  global_batch_size = per_replica_batch_size * _NUM_CORES.value
  data_config = configs.DatasetConfig(_DATASET.value)
  model_config = configs.ModelConfig(_MODEL_TYPE.value)

  train_steps_per_epoch = data_config.train_size // global_batch_size
  eval_steps_per_epoch = int(np.ceil(data_config.eval_size / global_batch_size))

  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=_TPU.value)
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)

  dataset = data_config.dataset_name_compat

  logging.info('Defining dataset.')
  data_train = obfuscations.ObfuscatedEmbeddingDataset(
      data_dir=_DATA_DIR_TRAIN.value,
      embed_dim=model_config.embed_dim,
      split='train',
      batch_size=per_replica_batch_size,
      num_views=len(_OBFUSCATIONS_TRAIN.value)
  )
  train_dataset = strategy.distribute_datasets_from_function(
      data_train.input_fn)

  data_test = obfuscations.ObfuscatedImageDataset(
      dataset,
      data_dir=_DATA_DIR_EVAL.value,
      obfuscation_list=[data_utils.CLEAN],
      split='test',
      batch_size=per_replica_batch_size)

  test_dataset = strategy.distribute_datasets_from_function(
      data_test.input_fn)

  logging.info('Defined dataset.')

  with strategy.scope():
    logging.info('Building model.')

    encoder = extended_model.IdentityEmbeddingMapper()

    model = extended_model.FeatureExtractor(
        model_link=model_config.model_link,
        bypass_base_model=(_INPUT_FEATURE_NAME.value == 'embed'),
        encoder=encoder,
        base_model_trainable=False
    )

    model_clf = extended_model.FeatureExtractorWithClassifier(
        num_classes=data_config.num_classes,
        feature_extractor=model,
        weight_decay=_WEIGHT_DECAY.value
    )

    if _LR_DECAY_TYPE.value == 'exponential':
      lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=_BASE_LR.value,
          decay_steps=train_steps_per_epoch * _LR_DECAY_EPOCHS.value,
          decay_rate=_LR_DECAY_FACTOR.value,
          staircase=True
      )
    elif _LR_DECAY_TYPE.value == 'cosine':
      lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
          initial_learning_rate=_BASE_LR.value,
          first_decay_steps=train_steps_per_epoch * _LR_DECAY_EPOCHS.value,
          alpha=1e-2
      )

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_scheduler,
        momentum=_MOMENTUM.value,
        nesterov=True)
    logging.info('Built model.')

    train_and_eval_model(
        model_clf=model_clf,
        train_dataset=train_dataset,
        train_steps_per_epoch=train_steps_per_epoch,
        test_dataset=test_dataset,
        eval_steps_per_epoch=eval_steps_per_epoch,
        optimizer=optimizer,
        strategy=strategy,
    )

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
