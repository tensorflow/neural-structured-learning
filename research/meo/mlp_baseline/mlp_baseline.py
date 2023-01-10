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

r"""Train a model mapping obfuscated image embeddings to clean image embeddings.

This code trains a model on top of a good feature extractor, in order to map
representations of obfuscated images to the representations of the corresponding
clean images. Training is performed by using a parallel dataset of images (a
clean version and an obfuscated one). The two different representations are then
matched via an MSE loss. At the same time, a classifier is built on top of the
reconstructed embeddings, in order to classify the obfuscated images correctly.
The base feature extractor can either be kept frozen (default behavior) or
finetuned as well during training.

This assumes that the input data has been modified, so that each item in the
dataset contains the following fields (where 'obf' is a given obfuscation):
- image_{obf}: the image corresponding to this particular obfuscation.

To launch this experiment, use xm_mlp_baseline.py. Example usage:

xmanager launch xm_mlp_baseline.py \
-- --xm_resource_alloc=group:<your allocation> \
--xm_resource_pool=<your resource pool>
--tpu_topology=<desired topology> \
--experiment_name=<name of experiment> \
--model_base_dir=<directory to save model> \
--data_dir=<director of training data>

TODO(smyrnisg): Move image normalization to the data pipeline.
"""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from meo.mlp_baseline import configs
from meo.mlp_baseline import data_utils
from meo.mlp_baseline import extended_model
from meo.mlp_baseline import losses as losses_lib
from meo.mlp_baseline import obfuscations
import numpy as np
import tensorflow as tf


_EPOCHS = flags.DEFINE_integer('epochs', 100, 'Number of training epochs.')

_TPU = flags.DEFINE_string('tpu', None, 'TPU to use.')
_BASE_LR = flags.DEFINE_float('base_lr', 1e-2, 'Learning rate.')
_LR_DECAY_FACTOR = flags.DEFINE_float(
    'lr_decay_factor', 0.1, 'Factor with which to decay learning rate per decay'
    'epoch.')
_LR_DECAY_EPOCHS = flags.DEFINE_integer(
    'lr_decay_epochs', 30, 'Number of epochs per learning rate decay step.')
_FINETUNE = flags.DEFINE_bool(
    'finetune', False, 'Choose whether to finetune the base model.')
_MOMENTUM = flags.DEFINE_float('momentum', 0.9, 'Momentum.')
_WEIGHT_DECAY = flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay.')

_EMBED_MAPPING_TYPE = flags.DEFINE_enum(
    'embed_mapping_type', 'MLP', ['MLP', 'VAE'],
    'Embedding mapping model type.')

_MLP_SIZES = flags.DEFINE_list(
    'mlp_sizes', ['1024', '512', '256', '512', '1024'],
    'List of layer sizes to be used in the our mappings. In the case of MLP, '
    'this defines the entire mapping. In the case of VAE, this list must '
    'contain an odd number of elements - the middle one is the latent dimension'
    ' and the rest correspond to the encoder and the decoder, respectively.')

_KL_LIM = flags.DEFINE_integer(
    'kl_lim', 20, 'Number of epochs during which to increase effect of KL '
    'divergence in the VAE model.')

_DATASET = flags.DEFINE_enum(
    'dataset', 'imagenet', ['imagenet', 'coco'],
    'Dataset to be used for the experiment.')

_EMBED_LOSS_WEIGHT = flags.DEFINE_float(
    'embed_loss_weight', 1.0, 'Factor with which to multiply embedding loss.')

_OBFUSCATIONS_TRAIN = flags.DEFINE_list(
    'obfuscations_train', data_utils.TRAIN_OBFUSCATIONS,
    'List of obfuscations to apply to the training data.')
_OBFUSCATIONS_EVAL = flags.DEFINE_list(
    'obfuscations_eval', data_utils.EVAL_OBFUSCATIONS,
    'List of obfuscations to apply to the evaluation data.')

_NUM_CORES = flags.DEFINE_integer('num_cores', None, 'Number of TPU cores.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'Batch size (per core).')
_DATA_DIR = flags.DEFINE_string('data_dir', None, 'Path to data.')
_MODEL_DIR = flags.DEFINE_string('model_dir', None, 'Path to model.')

_MODEL_TYPE = flags.DEFINE_string('model_type', 'resnet50',
                                  'Base model extractor.')

_NUM_SAVE_IMG = flags.DEFINE_integer('num_save_img', 1,
                                     'Number images to save in summary.')
_LOGGING_STEP = flags.DEFINE_integer('logging_step', 20,
                                     'Interval of steps between log messages.')


def save_sample_images(dataset: tf.distribute.DistributedDataset,
                       summary_writer: tf.summary.SummaryWriter,
                       strategy: tf.distribute.Strategy) -> None:
  """Save some sample images from the dataset to a TensorBoard summary.

  Args:
    dataset: Dataset from which to retrieve images.
    summary_writer: Summary writer object to log information. This will save the
      images to the corresponding TensorBoard summary.
    strategy: Strategy used for distributed training and evaluation.
  """
  iterator = iter(dataset)
  logging.info('Saving sample images.')
  for i in range(_NUM_SAVE_IMG.value):
    items = next(iterator)
    items = strategy.experimental_local_results(items)[0]
    logging.info('Loaded data.')

    image_set = [
        items['image_{}'.format(obf)][0, ...]
        for obf in _OBFUSCATIONS_TRAIN.value
    ]

    with summary_writer.as_default():
      tf.summary.image(
          'Train Image {}'.format(i), image_set, step=i)
    logging.info('Saved image %s.', i)


def train_and_eval_model(
    model_clf: extended_model.FeatureExtractorWithClassifier,
    train_dataset: tf.distribute.DistributedDataset,
    train_steps_per_epoch: int,
    test_dataset: tf.distribute.DistributedDataset,
    eval_steps_per_epoch: int,
    optimizer: tf.keras.optimizers.Optimizer,
    strategy: tf.distribute.Strategy) -> None:
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

    acc_names = ['clf_train_acc']
    for obf in _OBFUSCATIONS_EVAL.value:
      acc_names.append('clf_test_acc_{}'.format(obf))
    acc_metrics = {
        acc_name:
        tf.keras.metrics.SparseCategoricalAccuracy(acc_name, dtype=tf.float32)
        for acc_name in acc_names
    }

    clf_checkpoint = tf.train.Checkpoint(model=model_clf, optimizer=optimizer)

  clf_train_summary_writer = tf.summary.create_file_writer(
      os.path.join(
          _MODEL_DIR.value,
          'summaries'
          f'lr_{_BASE_LR.value}_decay_{_WEIGHT_DECAY.value}',
          'clf_train'
      )
  )

  clf_test_summary_writer = tf.summary.create_file_writer(
      os.path.join(
          _MODEL_DIR.value,
          'summaries',
          f'lr_{_BASE_LR.value}_decay_{_WEIGHT_DECAY.value}',
          'clf_test'
      )
  )

  @tf.function
  def clf_train_step(iterator):

    def step_fn(inputs):
      clean_images = inputs['image_Clean']
      all_obfs = tf.stack(
          [inputs['image_{}'.format(obf)] for obf in _OBFUSCATIONS_TRAIN.value],
          axis=1)

      # Randomly pick an obfuscation for each image - the first obfuscation is
      # the clean obfuscation.
      indices = tf.random.uniform(
          shape=[all_obfs.shape[0]],
          minval=1,
          maxval=len(_OBFUSCATIONS_TRAIN.value),
          dtype=tf.int32)
      obf_images = tf.gather(all_obfs, indices, batch_dims=1)
      labels = inputs['label']

      if clean_images.dtype == tf.uint8:
        clean_images = tf.cast(clean_images, tf.float32) / 255.0
      if obf_images.dtype == tf.uint8:
        obf_images = tf.cast(obf_images, tf.float32) / 255.0

      with tf.GradientTape() as tape:
        clean_embeddings = model_clf.feature_extractor.encode_clean(
            clean_images, training=True)
        if _EMBED_MAPPING_TYPE.value == 'MLP':
          obf_embeddings = model_clf.feature_extractor(obf_images,
                                                       training=True)
          # Embedding matching loss.
          embed_loss = tf.reduce_mean(
              tf.keras.metrics.mean_squared_error(clean_embeddings,
                                                  obf_embeddings)
          )
        elif _EMBED_MAPPING_TYPE.value == 'VAE':
          obf_embeddings, z_mean, z_log_var = model_clf.feature_extractor(
              obf_images, training=True)

          # Calculate appropriate KL term weight (based on a sigmoid, which
          # ramps up to approximately 1 in _KL_LIM epochs). More specifically,
          # this sigmoid is chosen so that it spans the range [exp(-5), exp(5)],
          # approximately [0,1], within _KL_LIM epochs. See Figure 2 of
          # https://arxiv.org/pdf/1511.06349.pdf.
          kl_weight = tf.sigmoid(
              10.0 *
              (tf.cast(optimizer.iterations, tf.float32) -
               0.5 * tf.cast(_KL_LIM.value * train_steps_per_epoch, tf.float32))
              / tf.cast(_KL_LIM.value * train_steps_per_epoch, tf.float32))

          # Reconstruction loss and KL divergence loss for the VAE model.
          recon_loss = tf.reduce_mean(
              tf.keras.metrics.mean_squared_error(clean_embeddings,
                                                  obf_embeddings))
          kl_loss = losses_lib.vae_kl_divergence(z_mean, z_log_var)
          embed_loss = recon_loss + kl_weight * kl_loss

        # Crossentropy loss for the classifier.
        logits = model_clf.clf_layer(obf_embeddings, training=True)
        crossentropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

        # model_clf.losses contains the regularizer for the linear classifier.
        regularizer_loss = tf.reduce_sum(model_clf.losses)

        loss = _EMBED_LOSS_WEIGHT.value * embed_loss + crossentropy_loss + regularizer_loss

        # Divide by number of replicas to balance out the artificially increased
        # batch size.
        per_replica_loss = loss / strategy.num_replicas_in_sync

      gradients = tape.gradient(per_replica_loss, model_clf.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model_clf.trainable_variables))
      loss_metrics['clf_train_loss'].update_state(loss)
      loss_metrics['clf_train_embed_loss'].update_state(embed_loss)
      loss_metrics['clf_train_classification_loss'].update_state(
          crossentropy_loss)
      loss_metrics['clf_train_reg_loss'].update_state(regularizer_loss)
      acc_metrics['clf_train_acc'].update_state(labels, tf.nn.softmax(logits))

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def clf_test_step(item):

    def step_fn(inputs):
      labels = inputs['label']
      for obf in _OBFUSCATIONS_EVAL.value:
        images = inputs['image_{}'.format(obf)]
        if images.dtype == tf.uint8:
          images = tf.cast(images, tf.float32) / 255.0

        logits = model_clf(images, training=False)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        loss = tf.math.divide_no_nan(
            tf.reduce_sum(losses), tf.cast(tf.size(losses), dtype=tf.float32))

        loss_metrics['clf_test_loss_{}'.format(obf)].update_state(loss)
        acc_metrics['clf_test_acc_{}'.format(obf)].update_state(
            labels, tf.nn.softmax(logits))

    strategy.run(step_fn, args=(item,))

  save_sample_images(train_dataset, clf_train_summary_writer, strategy)
  train_iterator = iter(train_dataset)
  for epoch in range(_EPOCHS.value):
    logging.info('Training Epoch: %s', epoch)
    with clf_train_summary_writer.as_default():
      for step in range(train_steps_per_epoch):
        if step % _LOGGING_STEP.value == 0:
          logging.info('Training Epoch %s, Step %s', epoch, step)
        clf_train_step(train_iterator)

      logging.info(
          'Train Loss: %s, Train Accuracy %s %%',
          round(loss_metrics['clf_train_loss'].result().numpy(), 3),
          round(acc_metrics['clf_train_acc'].result().numpy() * 100.0, 3))

      for loss_name, loss in loss_metrics.items():
        if 'train' in loss_name:
          tf.summary.scalar(loss_name, loss.result().numpy(),
                            step=optimizer.iterations)
          loss.reset_state()

      lr_sched = tf.keras.optimizers.schedules.ExponentialDecay.from_config(
          optimizer.get_config()['learning_rate']['config']
      )
      tf.summary.scalar(
          'clf_learning_rate',
          lr_sched(optimizer.iterations).numpy(),
          step=optimizer.iterations)

      tf.summary.scalar(
          'clf_train_acc',
          acc_metrics['clf_train_acc'].result().numpy(),
          step=optimizer.iterations)
      acc_metrics['clf_train_acc'].reset_state()

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
  # TODO(smyrnisg) Change the following:
  # The obfuscation is set to 'Clean' simply for compatibility.
  data_train = obfuscations.ObfuscatedImageDataset(
      dataset,
      data_dir=_DATA_DIR.value,
      obfuscation_list=[data_utils.CLEAN],
      split='train',
      batch_size=per_replica_batch_size)

  train_dataset = strategy.distribute_datasets_from_function(
      data_train.input_fn)

  data_test = obfuscations.ObfuscatedImageDataset(
      dataset,
      data_dir=_DATA_DIR.value,
      obfuscation_list=[data_utils.CLEAN],
      split='test',
      batch_size=per_replica_batch_size)

  test_dataset = strategy.distribute_datasets_from_function(
      data_test.input_fn)

  logging.info('Defined dataset.')

  with strategy.scope():
    logging.info('Building model.')
    mlp_sizes = [
        int(_MLP_SIZES.value[i]) for i in range(len(_MLP_SIZES.value))
    ]
    if _EMBED_MAPPING_TYPE.value == 'MLP':
      encoder = extended_model.MLPEmbeddingMapper(
          embed_dim=model_config.embed_dim,
          mlp_sizes=mlp_sizes,
          weight_decay=_WEIGHT_DECAY.value)
    elif _EMBED_MAPPING_TYPE.value == 'VAE':
      encoder = extended_model.VAEEmbeddingMapper(
          mlp_sizes=mlp_sizes,
          embed_dim=model_config.embed_dim,
          weight_decay=_WEIGHT_DECAY.value)
    else:
      raise ValueError('Mapping type is not defined: {}'.format(
          _EMBED_MAPPING_TYPE.value))

    model = extended_model.FeatureExtractor(
        model_link=model_config.model_link,
        encoder=encoder,
        base_model_trainable=_FINETUNE.value)

    model_clf = extended_model.FeatureExtractorWithClassifier(
        num_classes=data_config.num_classes, feature_extractor=model)

    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=_BASE_LR.value,
        decay_steps=train_steps_per_epoch * _LR_DECAY_EPOCHS.value,
        decay_rate=_LR_DECAY_FACTOR.value,
        staircase=True
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
        strategy=strategy)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
