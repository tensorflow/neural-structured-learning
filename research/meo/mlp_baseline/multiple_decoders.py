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

r"""Train a model generating obfuscated embeddings along with a classifier.

This code trains a model which generates obfuscated embeddings, given the clean
embeddings of an image. The code makes use of an autoencoder architecture, with
a single encoder mapping clean embeddings to a latent space, and multiple
decoders mapping from the latent space to each obfuscation type. This
autoencoder architecture is discarded during inference.

These generated embeddings are used in conjunction with the embeddings of the
real obfuscated images to train a classifier on a given dataset. This classifier
is trained jointly with the model that generates embeddings for the obfuscated
images. During training, the loss due to the generated obfuscated embeddings is
weighted, starting from near 0 and ramping up to near 1 after a set amount of
epochs. During inference, as the autoencoder architecture is discarded, the
classifier is provided with embeddings computed from the feature extractor
model.


The input data to this model can be in either pixel space (original images) or
embedding space (precomputed outputs of a given model).

This assumes that the input data has been modified, so that each item in the
dataset contains the following fields (where 'obf' is a given obfuscation):
- image_{obf}: the image corresponding to this particular obfuscation.

To run this experiment, use the following xmanager script:
xmanager launch xm_multiple_decoders.py \
-- --xm_resource_alloc=group:<your allocation> \
--xm_resource_pool=<your resource pool>
--tpu_topology=<desired topology> \
--experiment_name=<name of experiment> \
--model_base_dir=<directory to save model> \
--data_dir_train=<directory of training data> \
--data_dir_eval=<directory of eval data>

TODO(smyrnisg): Move image normalization to the data pipeline.
"""

import os
from typing import Optional, Sequence

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
import tensorflow_hub as hub

_EPOCHS = flags.DEFINE_integer('epochs', 100, 'Number of training epochs.')

_TPU = flags.DEFINE_string('tpu', None, 'TPU to use.')
_BASE_LR = flags.DEFINE_float('base_lr', 1e-1, 'Initial learning rate.')
_LR_DECAY_TYPE = flags.DEFINE_enum(
    'lr_decay_type', 'exponential', ['cosine', 'exponential'],
    'Type of learning rate decay.')
_LR_DECAY_FACTOR = flags.DEFINE_float(
    'lr_decay_factor', 0.5, 'Factor with which to decay learning rate per decay'
    'epoch.')
_LR_DECAY_EPOCHS = flags.DEFINE_integer(
    'lr_decay_epochs', 30, 'Number of epochs per learning rate decay step.')
_FINETUNE = flags.DEFINE_bool(
    'finetune', False, 'Choose whether to finetune the base model.')

_MOMENTUM = flags.DEFINE_float('momentum', 0.9, 'Momentum.')
_WEIGHT_DECAY = flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay.')

_MLP_SIZES = flags.DEFINE_list(
    'mlp_sizes', ['1024', '512', '256', '512', '1024'],
    'List of layer sizes to be used in the our mappings. In the case of MLP, '
    'this defines the entire mapping. In the case of VAE, this list must '
    'contain an odd number of elements - the middle one is the latent dimension'
    ' and the rest correspond to the encoder and the decoder, respectively.')

_EPOCH_LIM = flags.DEFINE_integer(
    'epoch_lim', 20, 'Number of epochs during which to ramp up the weight '
    'assigned to the classification loss of the generated obfuscated samples.')

_SKIP_CONNECTION = flags.DEFINE_bool(
    'skip_connection', True, 'Whether to use a skip connection.')

_DATASET = flags.DEFINE_enum(
    'dataset', 'imagenet', ['imagenet', 'coco'],
    'Dataset to be used for the experiment.')

_INPUT_FEATURE_NAME = flags.DEFINE_enum(
    'input_feature_name', 'embed', ['pixel', 'embed'],
    'Format of input data (choice between pixel i.e. raw images and embed i.e.'
    'representations produced by target model'
)

_EMBED_LOSS_WEIGHT = flags.DEFINE_float(
    'embed_loss_weight', 5.0, 'Factor with which to multiply embedding loss.')

_EMBED_RECON_LOSS = flags.DEFINE_enum(
    'embed_recon_loss', 'Contrastive', ['MSE', 'Contrastive'],
    'Loss to be used to match the embeddings.'
)

_OBFUSCATIONS_TRAIN = flags.DEFINE_list(
    'obfuscations_train', data_utils.TRAIN_OBFUSCATIONS,
    'List of obfuscations to apply to the training data.')
_OBFUSCATIONS_EVAL = flags.DEFINE_list(
    'obfuscations_eval', data_utils.EVAL_OBFUSCATIONS,
    'List of obfuscations to apply to the evaluation data.')

_NUM_CORES = flags.DEFINE_integer('num_cores', None, 'Number of TPU cores.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 64, 'Batch size (per core).')
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

_USE_TEXT_ENCODER = flags.DEFINE_bool('use_text_encoder', False,
                                      'Whether to use text encoder.')

_SAVED_MODEL = flags.DEFINE_string(
    'saved_model', None, 'Path to saved model for custom pretraining.')


def sigmoid_with_limit(t: int, limit: int) -> float:
  """Calculate a sigmoid that ramps up to 1 after a certain limit.

  This function calculates a sigmoid function, defined as `sigmoid(x)`, where
  `x = 10 * t / limit - 5`. This results in `x` ranging from -5 at 0 to 5 at
  `limit`, and thus the output starts from a value near 0 at `t = 0` and ramps
  up near 1 at `t = limit`.

  Args:
    t: The point at which to evaluate the sigmoid.
    limit: The point at which the sigmoid must attain the value 1-exp(-5),
      which is close enough to 1.

  Returns:
    The value of the sigmoid at the chosen point.
  """
  result = tf.sigmoid(
      10.0 * tf.cast(t, tf.float32) / tf.cast(limit, tf.float32) - 5.0
  )
  return result


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
    strategy: tf.distribute.Strategy,
    gen_dataset: Optional[tf.distribute.DistributedDataset] = None,
    text_embeds: Optional[tf.Tensor] = None,
    custom_pretrain_model: Optional[extended_model.FeatureExtractor] = None
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
    gen_dataset: Optional dataset of pre-generated images.
    text_embeds: Optionally use text embeddings for the obfuscations.
    custom_pretrain_model: If present, a custom pretrained model will be used
      for the embeddings, instead of one on tfhub.
  """
  with strategy.scope():
    loss_names = [
        'clf_train_loss', 'clf_train_classification_loss', 'clf_train_reg_loss',
        'clf_train_embed_loss', 'clf_train_text_embed_loss'
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
  def clf_train_step(item, item_gen=None):

    def step_fn(inputs, gen_inputs=None):
      if _INPUT_FEATURE_NAME.value == 'pixel':
        clean_data = inputs['image_Clean']
        obfuscated_data = tf.stack([
            inputs['image_{}'.format(obf)]
            for obf in _OBFUSCATIONS_TRAIN.value[1:]
        ], axis=1)
        labels = inputs['label']

        if clean_data.dtype == tf.uint8:
          clean_data = tf.cast(clean_data, tf.float32) / 255.0
        if obfuscated_data.dtype == tf.uint8:
          obfuscated_data = tf.cast(obfuscated_data, tf.float32) / 255.0

      elif _INPUT_FEATURE_NAME.value == 'embed':
        clean_data = tf.squeeze(inputs[0][:, 0, :])
        obfuscated_data = inputs[0][:, 1:, :]
        labels = inputs[1]

      if gen_inputs:
        gen_images = gen_inputs[0]
        gen_labels = gen_inputs[1]
        extra_generated_embed = model_clf.feature_extractor.base_model(
            gen_images, training=False)
      else:
        extra_generated_embed = None

      with tf.GradientTape() as tape:
        if _INPUT_FEATURE_NAME.value == 'pixel':
          if custom_pretrain_model:
            clean_embed = custom_pretrain_model(clean_data, training=False)
          else:
            clean_embed = model_clf.feature_extractor.encode_clean(
                clean_data, training=True)

          # Pack the images from the shape of [batch_size, num_views, H, W, C]
          # to [batch_size * num_views, H, W, C] to compute embeddings from the
          # encoder model, and reshape the output embeddings back to
          # [batch_size, num_views, embed_dim].
          orig_shape = tf.shape(obfuscated_data)
          num_views = orig_shape[1]
          height = orig_shape[2]
          width = orig_shape[3]
          channels = orig_shape[4]
          obfuscated_data = tf.reshape(
              obfuscated_data, [-1, height, width, channels]
          )
          if custom_pretrain_model:
            real_obfuscated_embed = custom_pretrain_model(
                obfuscated_data, training=False)
          else:
            real_obfuscated_embed = model_clf.feature_extractor.encode_clean(
                obfuscated_data, training=True)
          real_obfuscated_embed = tf.reshape(
              real_obfuscated_embed,
              [-1, num_views, tf.shape(clean_embed)[-1]]
          )
        else:
          clean_embed = clean_data
          real_obfuscated_embed = obfuscated_data
          orig_shape = tf.shape(obfuscated_data)
          num_views = orig_shape[1]

        generated_obfuscated_embed = model_clf.feature_extractor(
            clean_data, training=True)

        # Embedding loss for the autoencoder.
        embed_loss = losses_lib.reconstruction_loss(
            real_obfuscated_embed,
            generated_obfuscated_embed,
            loss_type=_EMBED_RECON_LOSS.value
        )

        # If using text embedding, add an extra loss term
        if _USE_TEXT_ENCODER.value:
          all_text_embeds = tf.tile(
              tf.expand_dims(text_embeds[1][1:], axis=0), [orig_shape[0], 1, 1])
          text_embed_loss = losses_lib.reconstruction_loss(
              all_text_embeds,
              generated_obfuscated_embed,
              loss_type=_EMBED_RECON_LOSS.value
          )
        else:
          text_embed_loss = 0

        # Reshape the embeddings to [batch_size * num_views,  embed_dim] for
        # classification.
        real_obfuscated_embed = tf.reshape(
            real_obfuscated_embed,
            [-1, tf.shape(real_obfuscated_embed)[-1]]
        )
        generated_obfuscated_embed = tf.reshape(
            generated_obfuscated_embed,
            [-1, tf.shape(generated_obfuscated_embed)[-1]]
        )
        if gen_inputs:
          generated_obfuscated_embed = tf.concat(
              [generated_obfuscated_embed, extra_generated_embed], axis=0)

        # Crossentropy loss for the classifier.
        real_logits = model_clf.clf_layer(
            real_obfuscated_embed, training=True
        )
        generated_logits = model_clf.clf_layer(
            generated_obfuscated_embed, training=True
        )

        # Multiplier for the generated sample loss, ramping up as training
        # progresses.
        loss_ramp_weight = sigmoid_with_limit(
            optimizer.iterations, _EPOCH_LIM.value * train_steps_per_epoch
        )

        # Duplicate the labels by the number of views.
        labels = tf.tile(tf.expand_dims(labels, axis=-1), [1, num_views])
        labels = tf.reshape(labels, [-1])

        if gen_inputs:
          all_labels = tf.concat([labels, gen_labels], axis=0)
        else:
          all_labels = labels

        crossentropy_loss = losses_lib.weighted_crossentropy_loss(
            labels,
            real_logits,
            generated_logits,
            loss_ramp_weight,
            gen_labels=all_labels
        )

        # model_clf.losses contains the regularizer for the linear classifier.
        regularizer_loss = tf.reduce_sum(model_clf.losses)

        # loss = embed_loss + crossentropy_loss + regularizer_loss
        loss = _EMBED_LOSS_WEIGHT.value * (
            embed_loss + text_embed_loss) + crossentropy_loss + regularizer_loss

        # Divide by number of replicas to balance out the artificially increased
        # batch size.
        per_replica_loss = loss / strategy.num_replicas_in_sync

      gradients = tape.gradient(per_replica_loss, model_clf.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model_clf.trainable_variables))
      loss_metrics['clf_train_loss'].update_state(loss)
      loss_metrics['clf_train_embed_loss'].update_state(embed_loss)
      loss_metrics['clf_train_text_embed_loss'].update_state(text_embed_loss)
      loss_metrics['clf_train_classification_loss'].update_state(
          crossentropy_loss)
      loss_metrics['clf_train_reg_loss'].update_state(regularizer_loss)

    strategy.run(
        step_fn,
        args=(item, item_gen),
    )

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

  if _INPUT_FEATURE_NAME.value == 'pixel':
    save_sample_images(train_dataset, clf_train_summary_writer, strategy)

  train_iterator = iter(train_dataset)
  gen_iterator = iter(gen_dataset) if gen_dataset else None
  for epoch in range(initial_epoch, _EPOCHS.value):
    logging.info('Training Epoch: %s', epoch)
    with clf_train_summary_writer.as_default():
      for step in range(train_steps_per_epoch):
        if step % _LOGGING_STEP.value == 0:
          logging.info('Training Epoch %s, Step %s', epoch, step)
        item = next(train_iterator)
        item_gen = next(gen_iterator) if gen_iterator else None
        clf_train_step(item, item_gen)

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
  if _INPUT_FEATURE_NAME.value == 'pixel':
    # TODO(smyrnisg) The obfuscation_list option below is obsolete - it is set
    # to ['Clean'] just to preserve compatibility with obfuscations.py, but the
    # current code assumes that all the obfuscations can be found in one file.
    # This inconsistency should be removed.
    data_train = obfuscations.ObfuscatedImageDataset(
        dataset,
        data_dir=_DATA_DIR_TRAIN.value,
        obfuscation_list=[data_utils.CLEAN],
        split='train',
        batch_size=per_replica_batch_size)
  elif _INPUT_FEATURE_NAME.value == 'embed':
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

  # If using text data, find the embeddings of the obfuscation captions.
  if _USE_TEXT_ENCODER.value:
    text_model = tf.keras.Sequential([
        hub.KerasLayer(configs.ModelConfig('align_text').model_link)
    ])
    text_model.compile(optimizer='Adam', loss='mse')
    obfuscation_texts = configs.OBFUSCATION_CAPTIONS
    text_embeds = text_model.predict(obfuscation_texts)
  else:
    text_embeds = None

  logging.info('Defined dataset.')

  with strategy.scope():
    logging.info('Building model.')
    mlp_sizes = [
        int(_MLP_SIZES.value[i]) for i in range(len(_MLP_SIZES.value))
    ]

    encoder = extended_model.AutoEncoderEmbeddingMapper(
        mlp_sizes=mlp_sizes,
        embed_dim=model_config.embed_dim,
        num_decoders=(len(_OBFUSCATIONS_TRAIN.value)-1),
        skip_connection=_SKIP_CONNECTION.value,
        weight_decay=_WEIGHT_DECAY.value
    )

    model = extended_model.FeatureExtractor(
        model_link=model_config.model_link,
        bypass_base_model=(_INPUT_FEATURE_NAME.value == 'embed'),
        encoder=encoder,
        base_model_trainable=_FINETUNE.value
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

    # Add an extra model to retrieve embeddings
    if _SAVED_MODEL.value:
      temp_model = extended_model.FeatureExtractor(
          model_link=model_config.model_link,
          bypass_base_model=(_INPUT_FEATURE_NAME.value == 'embed'),
          encoder=extended_model.IdentityEmbeddingMapper(),
          base_model_trainable=_FINETUNE.value
      )

      temp_model_clf = extended_model.FeatureExtractorWithClassifier(
          num_classes=data_config.num_classes,
          feature_extractor=temp_model,
          weight_decay=_WEIGHT_DECAY.value
      )

      temp_checkpoint = tf.train.Checkpoint(model=temp_model_clf)
      latest_checkpoint = tf.train.latest_checkpoint(_SAVED_MODEL.value)
      if latest_checkpoint:
        temp_checkpoint.restore(latest_checkpoint)
      else:
        raise ValueError('No custom pretrained model.')
    else:
      temp_model_clf = None

    train_and_eval_model(
        model_clf=model_clf,
        train_dataset=train_dataset,
        train_steps_per_epoch=train_steps_per_epoch,
        test_dataset=test_dataset,
        eval_steps_per_epoch=eval_steps_per_epoch,
        optimizer=optimizer,
        strategy=strategy,
        text_embeds=text_embeds,
        custom_pretrain_model=temp_model_clf.feature_extractor
        if temp_model_clf else None)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
