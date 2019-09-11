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
"""Run script for training Graph Agreement Models on MNIST and other datasets.

Throughout this file, the suffix "_cls" refers to the classification model, and
"_agr" to the agreement model.

The supported datasets are the following tensorflow_datasets:
mnist, cifar10, cifar100, svhn_cropped, fashion_mnist.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from logging import config

import os
from absl import app
from absl import flags

from gam.data.dataset import Dataset
from gam.data.loaders import load_data_realistic_ssl
from gam.data.loaders import load_data_tf_datasets
from gam.models.cnn import ImageCNNAgreement
from gam.models.mlp import MLP
from gam.models.wide_resnet import WideResnet
from gam.trainer.trainer_cotrain import TrainerCotraining
import numpy as np
import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'dataset_name', '',
    'Dataset name. Supported options are: mnist, cifar10, cifar100, '
    'svhn_cropped, fashion_mnist.')
flags.DEFINE_string(
    'data_source', 'tensorflow_datasets', 'Data source. Valid options are: '
    '`tensorflow_datasets`, `realistic_ssl`')
flags.DEFINE_integer(
    'target_num_train_per_class', 400,
    'Number of samples per class to use for training.')
flags.DEFINE_integer(
    'target_num_val', 1000,
    'Number of samples to be used for validation.')
flags.DEFINE_integer(
    'seed', 123,
    'Seed used by the random number generators.')
flags.DEFINE_bool(
    'load_preprocessed', False,
    'Specifies whether to load data already preprocessed. If False, it reads'
    'the original data and splits it.')
flags.DEFINE_bool(
    'save_preprocessed', False,
    'Specifies whether the preprocessed should be saved to pickle.')
flags.DEFINE_string(
    'filename_preprocessed_data', 'preprocessed_data.pickle',
    'Name of the pickle file where the preprocessed data will be loaded from '
    'or stored.')
flags.DEFINE_string(
    'label_map_path', '',
    'Path to the json files containing the label sample indices for '
    'Realistic SSL.')
flags.DEFINE_string(
    'data_output_dir', './outputs',
    'Path to a folder where to save the preprocessed dataset.')
flags.DEFINE_string(
    'output_dir', './outputs',
    'Path to a folder where checkpoints, summaries and other outputs are '
    'stored.')
flags.DEFINE_string(
    'logging_config', '', 'Path to logging configuration file.')
flags.DEFINE_string(
    'model_cls', 'mlp',
    'Model type for the classification model. '
    'Options are: `mlp`, `cnn`, `wide_resnet`')
flags.DEFINE_string(
    'model_agr', 'mlp',
    'Model type for the agreement model. Options are: `mlp`, `cnn`, '
    '`wide_resnet`.')
flags.DEFINE_float(
    'learning_rate_cls', 0.001,
    'Initial learning rate of the classification model.')
flags.DEFINE_float(
    'learning_rate_agr', 0.001,
    'Initial learning rate of the agreement model.')
flags.DEFINE_float(
    'learning_rate_decay_cls', None,
    'Learning rate decay factor for the classification model.')
flags.DEFINE_float(
    'learning_rate_decay_agr', None,
    'Learning rate decay factor for the agreement model.')
flags.DEFINE_float(
    'lr_decay_rate_cls', None,
    'Learning rate decay rate for the classification model.')
flags.DEFINE_integer(
    'lr_decay_steps_cls', None,
    'Learning rate decay steps for the classification model.')
flags.DEFINE_float(
    'lr_decay_rate_agr', None,
    'Learning rate decay rate for the agreement model.')
flags.DEFINE_integer(
    'lr_decay_steps_agr', None,
    'Learning rate decay steps for the agreement model.')
flags.DEFINE_integer(
    'num_epochs_per_decay_cls', 350,
    'Number of epochs after which the learning rate decays for the '
    'classification model.')
flags.DEFINE_integer(
    'num_epochs_per_decay_agr', 350,
    'Number of epochs after which the learning rate decays for the '
    'agreement model.')
flags.DEFINE_integer(
    'max_num_iter_cotrain', 100, 'Number of epochs to train.')
flags.DEFINE_integer(
    'min_num_iter_cls', 200, 'Minimum number of epochs to train for.')
flags.DEFINE_integer(
    'max_num_iter_cls', 100000, 'Maximum number of epochs to train for.')
flags.DEFINE_integer(
    'num_iter_after_best_val_cls', 2000,
    'Minimum number of iterations to train the classification model for after '
    'the best validation accuracy is improved.')
flags.DEFINE_integer(
    'min_num_iter_agr', 200,
    'Minimum number of iterations to train the agreement model for.')
flags.DEFINE_integer(
    'max_num_iter_agr', 100000,
    'Maximum number of iterations to train the agreement model for.')
flags.DEFINE_integer(
    'num_iter_after_best_val_agr', 5000,
    'Minimum number of iterations to train the agreement model for after '
    'the best validation accuracy is improved.')
flags.DEFINE_integer(
    'num_samples_to_label', 500,
    'Number of samples to label after each co-train iteration.')
flags.DEFINE_float(
    'min_confidence_new_label', 0.4,
    'Minimum confidence required for a sample to be added to the labeled set.')
flags.DEFINE_bool(
    'keep_label_proportions', True,
    'Whether the newly labeled nodes should have the same label proportions'
    'as the original labeled data.')
flags.DEFINE_integer(
    'num_warm_up_iter_agr', 1,
    'Minimum number of co-train iterations the agreement must be trained '
    'before it is used in the classifier.')
flags.DEFINE_float(
    'ratio_valid_agr', 0.1,
    'Ratio of edges used for validating the agreement model.')
flags.DEFINE_integer(
    'max_samples_valid_agr', 10000,
    'Max number of samples to set aside for validating the agreement model.')
flags.DEFINE_string(
    'hidden_cls', '128_64_32',
    'String representing the number of units of the hidden layers of the '
    'classification model. This is encoded as a sequence of numbers separated '
    'by underscores (e.g., `128_64_32`), where each number is the number of '
    'units in a layer, counting from the inputs towards outputs')
flags.DEFINE_string(
    'hidden_agr', '128_64_32',
    'String representing the number of units of the hidden layers of the '
    'agreement model.')
flags.DEFINE_string(
    'hidden_aggreg', '',
    'String representing the number of units of the hidden layers of the '
    'aggregation network of the agreement model.')
flags.DEFINE_float(
    'weight_decay_cls', 0,
    'Weight of the L2 penalty on the classification model weights.')
flags.DEFINE_string(
    'weight_decay_schedule_cls', None,
    'Schedule for decaying the weight decay in the classification model. '
    'Choose bewteen None or linear.')
flags.DEFINE_float(
    'weight_decay_agr', 0,
    'Weight of the L2 penalty on the agreement model weights.')
flags.DEFINE_string(
    'weight_decay_schedule_agr', None,
    'Schedule for decaying the weight decay in the agreement model. Choose '
    'between None or linear.')
flags.DEFINE_integer(
    'batch_size_agr', 512, 'Batch size for agreement model.')
flags.DEFINE_integer(
    'batch_size_cls', 512, 'Batch size for classification model.')
flags.DEFINE_float(
    'gradient_clip', None,
    'The gradient clipping global norm value. If None, no clipping is done.')
flags.DEFINE_integer(
    'logging_step_cls', 200,
    'Print summary of the classification model training every this number of '
    'iterations.')
flags.DEFINE_integer(
    'logging_step_agr', 200,
    'Print summary of the agreement model training every this number of '
    'iterations.')
flags.DEFINE_integer(
    'summary_step_cls', 100,
    'Print summary of classification model training every this number of '
    'iterations.')
flags.DEFINE_integer(
    'summary_step_agr', 100,
    'Print summary of the agreement model training every this number of '
    'iterations.')
flags.DEFINE_integer(
    'eval_step_cls', 100,
    'Evaluate classification model every this number of iterations.')
flags.DEFINE_integer(
    'eval_step_agr', 100,
    'Evaluate the agreement model every this number of iterations.')
flags.DEFINE_bool(
    'warm_start_cls', False,
    'Whether to reinitialize the parameters of the classification model before '
    'retraining (if False), or use the ones from the previous cotrain'
    ' iteration.')
flags.DEFINE_bool(
    'warm_start_agr', False,
    'Whether to reinitialize the parameters of the agreement model before '
    'retraining (if False), or use the ones from the previous cotrain '
    'iteration.')
flags.DEFINE_bool(
    'use_perfect_agreement', False, 'Whether to use perfect agreement.')
flags.DEFINE_bool(
    'use_perfect_classifier', False, 'Whether to use perfect classifier.')
flags.DEFINE_float(
    'reg_weight_ll', 0.00, 'Regularization weight for labeled-labeled edges.')
flags.DEFINE_float(
    'reg_weight_lu', 0.1, 'Regularization weight for labeled-unlabeled edges.')
flags.DEFINE_float(
    'reg_weight_uu', 0.05,
    'Regularization weight for unlabeled-unlabeled edges.')
flags.DEFINE_integer(
    'num_pairs_reg', 128,
    'Number of pairs of nodes to use in the agreement loss term of the '
    'classification model.')
flags.DEFINE_string(
    'aggregation_agr_inputs', 'dist',
    'Operation to apply on the pair of nodes in the agreement model. '
    'Available options are `add`, `dist`, `concat`, `project_add`,'
    '`project_dist`, `project_concat` and None.')
flags.DEFINE_bool(
    'penalize_neg_agr', True,
    'Whether to encourage differences when agreement is negative.')
flags.DEFINE_bool(
    'use_l2_cls', True,
    'Whether to use L2 loss for the classifier, not cross entropy.')
flags.DEFINE_bool(
    'first_iter_original', True,
    'Whether to use the original model in the first iteration, without self '
    'labeling or agreement loss.')
flags.DEFINE_bool(
    'inductive', True,
    'Whether to use an inductive or transductive SSL setting.')
flags.DEFINE_string(
    'experiment_suffix', '',
    'A suffix you might want to add at the end of the experiment name to'
    'identify it.')
flags.DEFINE_bool(
    'eval_acc_pred_by_agr', False,
    'Whether to compute the accuracy of a classification model that makes '
    'label predictions using the agreement model`s predictions. This is done'
    'by computing the majority vote of the labeled samples, weighted by the '
    ' agreement model. This is for monitoring the progress only.')
flags.DEFINE_integer(
    'num_neighbors_pred_by_agr', 50,
    'Number of labeled samples to use when predicting by agreement.')
flags.DEFINE_string(
    'optimizer', 'adam',
    'Which optimizer to use. Valid options are `adam`, `amsgrad`.')
flags.DEFINE_bool(
  'load_from_checkpoint', False,
  'Whether to load the data that has been self-labeled from a previous run, if '
  'available. This is useful if a process can get preempted or interrupted.')


def parse_layers_string(layers_string):
  """Convert a layer size string (e.g., `128_64_32`) to a list of integers."""
  if not layers_string:
    return ()
  num_hidden = layers_string.split('_')
  num_hidden = [int(num) for num in num_hidden]
  return num_hidden


def load_data():
  """Loads data."""
  if FLAGS.data_source == 'tensorflow_datasets':
    return load_data_tf_datasets(FLAGS.dataset_name,
                                 FLAGS.target_num_train_per_class,
                                 FLAGS.target_num_val,
                                 FLAGS.seed)
  elif FLAGS.data_source == 'realistic_ssl':
    return load_data_realistic_ssl(FLAGS.dataset_name,
                                   FLAGS.filename_preprocessed_data,
                                   FLAGS.label_map_path)
  raise ValueError('Unsupported dataset source name: %s' % FLAGS.data_source)


def pick_model(data):
  """Picks the models depending on the provided configuration flags."""
  # Create model classification.
  if FLAGS.model_cls == 'mlp':
    hidden_cls = (parse_layers_string(FLAGS.hidden_cls)
                      if FLAGS.hidden_cls is not None else [])
    model_cls = MLP(
        output_dim=data.num_classes,
        hidden_sizes=hidden_cls,
        activation=tf.nn.leaky_relu,
        name='mlp_cls')
  elif FLAGS.model_cls == 'cnn':
    if FLAGS.dataset_name in ('mnist', 'fashion_mnist'):
      channels = 1
    elif FLAGS.dataset_name in ('cifar10', 'cifar100', 'svhn_cropped', 'svhn'):
      channels = 3
    else:
      raise ValueError('Dataset name `%s` unsupported.' % FLAGS.dataset_name)
    model_cls = ImageCNNAgreement(
        output_dim=data.num_classes,
        channels=channels,
        activation=tf.nn.leaky_relu,
        name='cnn_cls')
  elif FLAGS.model_cls == 'wide_resnet':
    model_cls = WideResnet(
        num_classes=data.num_classes,
        lrelu_leakiness=0.1,
        horizontal_flip=FLAGS.dataset_name in ('cifar10',),
        random_translation=True,
        gaussian_noise=FLAGS.dataset_name not in ('svhn', 'svhn_cropped'),
        width=2,
        num_residual_units=4,
        name='wide_resnet_cls')
  else:
    raise NotImplementedError()

  # Create model agreement.
  hidden_agr = (parse_layers_string(FLAGS.hidden_agr)
                if FLAGS.hidden_agr is not None else [])
  hidden_aggreg = (parse_layers_string(FLAGS.hidden_aggreg)
                   if FLAGS.hidden_aggreg is not None else [])
  if FLAGS.model_agr == 'mlp':
    model_agr = MLP(
        output_dim=1,
        hidden_sizes=hidden_agr,
        activation=tf.nn.leaky_relu,
        aggregation=FLAGS.aggregation_agr_inputs,
        hidden_aggregation=FLAGS.hidden_aggreg,
        is_binary_classification=True,
        name='mlp_agr')
  elif FLAGS.model_agr == 'cnn':
    if FLAGS.dataset_name in ('mnist', 'fashion_mnist'):
      channels = 1
    elif FLAGS.dataset_name in ('cifar10', 'cifar100', 'svhn_cropped', 'svhn'):
      channels = 3
    else:
      raise ValueError('Dataset name `%s` unsupported.' % FLAGS.dataset_name)
    model_agr = ImageCNNAgreement(
        output_dim=1,
        channels=channels,
        activation=tf.nn.leaky_relu,
        aggregation=FLAGS.aggregation_agr_inputs,
        hidden_aggregation=hidden_aggreg,
        is_binary_classification=True,
        name='cnn_agr')
  elif FLAGS.model_agr == 'wide_resnet':
    model_agr = WideResnet(
        num_classes=1,
        lrelu_leakiness=0.1,
        horizontal_flip=FLAGS.dataset_name in ('cifar10',),
        random_translation=True,
        gaussian_noise=FLAGS.dataset_name not in ('svhn', 'svhn_cropped'),
        width=2,
        num_residual_units=4,
        name='wide_resnet_cls',
        is_binary_classification=True,
        aggregation=FLAGS.aggregation_agr_inputs,
        activation=tf.nn.leaky_relu,
        hidden_aggregation=hidden_aggreg)
  else:
    raise NotImplementedError()
  return model_cls, model_agr


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.logging_config:
    print('Setting logging configuration: ', FLAGS.logging_config)
    config.fileConfig(FLAGS.logging_config)

  # Set random seed.
  np.random.seed(FLAGS.seed)
  tf.set_random_seed(FLAGS.seed)

  # Potentially create a folder where to save the preprocessed data.
  if not os.path.exists(FLAGS.data_output_dir):
    os.makedirs(FLAGS.data_output_dir)

  # Load and potentially preprocess data.
  if FLAGS.load_preprocessed:
    logging.info('Loading preprocessed data...')
    path = os.path.join(FLAGS.data_output_dir, FLAGS.filename_preprocessed_data)
    data = Dataset.load_from_pickle(path)
  else:
    data = load_data()
    if FLAGS.save_preprocessed:
      assert FLAGS.output_dir
      path = os.path.join(FLAGS.data_output_dir,
                          FLAGS.filename_preprocessed_data)
      data.save_to_pickle(path)
      logging.info('Preprocessed data saved to %s.', path)

  # Put together parameters to create a model name.
  model_name = FLAGS.model_cls + (('_' + FLAGS.hidden_cls)
                                  if FLAGS.model_cls == 'mlp' else '')
  model_name += '-' + FLAGS.model_agr + (('_' + FLAGS.hidden_agr)
                                         if FLAGS.model_agr == 'mlp' else '')
  model_name += ('-aggr_' + FLAGS.aggregation_agr_inputs + '_' +
                 FLAGS.hidden_aggreg)
  model_name += ('-add_%d-conf_%.2f-iter_cls_%d-iter_agr_%d-batch_cls_%d' %
                 (FLAGS.num_samples_to_label, FLAGS.min_confidence_new_label,
                  FLAGS.max_num_iter_cls, FLAGS.max_num_iter_agr,
                  FLAGS.batch_size_cls))
  model_name += '-perfectAgr' if FLAGS.use_perfect_agreement else ''
  model_name += '-perfectCls' if FLAGS.use_perfect_classifier else ''
  model_name += '-keepProp' if FLAGS.keep_label_proportions else ''
  model_name += '-PenNegAgr' if FLAGS.penalize_neg_agr else ''
  model_name += '-inductive' if FLAGS.inductive else ''
  model_name += '-L2Loss' if FLAGS.use_l2_cls else '-CELoss'
  model_name += '-seed_' + str(FLAGS.seed)
  model_name += FLAGS.experiment_suffix
  logging.info('Model name: %s', model_name)

  # Create directories for model checkpoints, summaries, and
  # self-labeled data backup.
  summary_dir = os.path.join(FLAGS.output_dir, 'summaries', FLAGS.dataset_name,
                             model_name)
  checkpoints_dir = os.path.join(FLAGS.output_dir, 'checkpoints', model_name)
  data_dir = os.path.join(FLAGS.data_output_dir, 'data_checkpoints', model_name)
  if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  # Select the model based on the provided FLAGS.
  model_cls, model_agr = pick_model(data)

  # Train.
  trainer = TrainerCotraining(
      model_cls=model_cls,
      model_agr=model_agr,
      max_num_iter_cotrain=FLAGS.max_num_iter_cotrain,
      min_num_iter_cls=FLAGS.min_num_iter_cls,
      max_num_iter_cls=FLAGS.max_num_iter_cls,
      num_iter_after_best_val_cls=FLAGS.num_iter_after_best_val_cls,
      min_num_iter_agr=FLAGS.min_num_iter_agr,
      max_num_iter_agr=FLAGS.max_num_iter_agr,
      num_iter_after_best_val_agr=FLAGS.num_iter_after_best_val_agr,
      num_samples_to_label=FLAGS.num_samples_to_label,
      min_confidence_new_label=FLAGS.min_confidence_new_label,
      keep_label_proportions=FLAGS.keep_label_proportions,
      num_warm_up_iter_agr=FLAGS.num_warm_up_iter_agr,
      optimizer=tf.train.AdamOptimizer,
      gradient_clip=FLAGS.gradient_clip,
      batch_size_agr=FLAGS.batch_size_agr,
      batch_size_cls=FLAGS.batch_size_cls,
      learning_rate_cls=FLAGS.learning_rate_cls,
      learning_rate_agr=FLAGS.learning_rate_agr,
      enable_summaries=True,
      enable_summaries_per_model=True,
      summary_dir=summary_dir,
      summary_step_cls=FLAGS.summary_step_cls,
      summary_step_agr=FLAGS.summary_step_agr,
      logging_step_cls=FLAGS.logging_step_cls,
      logging_step_agr=FLAGS.logging_step_agr,
      eval_step_cls=FLAGS.eval_step_cls,
      eval_step_agr=FLAGS.eval_step_agr,
      checkpoints_dir=checkpoints_dir,
      checkpoints_step=1,
      data_dir=data_dir,
      abs_loss_chg_tol=1e-10,
      rel_loss_chg_tol=1e-7,
      loss_chg_iter_below_tol=30,
      use_perfect_agr=FLAGS.use_perfect_agreement,
      use_perfect_cls=FLAGS.use_perfect_classifier,
      warm_start_cls=FLAGS.warm_start_cls,
      warm_start_agr=FLAGS.warm_start_agr,
      ratio_valid_agr=FLAGS.ratio_valid_agr,
      max_samples_valid_agr=FLAGS.max_samples_valid_agr,
      weight_decay_cls=FLAGS.weight_decay_cls,
      weight_decay_schedule_cls=FLAGS.weight_decay_schedule_cls,
      weight_decay_schedule_agr=FLAGS.weight_decay_schedule_agr,
      weight_decay_agr=FLAGS.weight_decay_agr,
      reg_weight_ll=FLAGS.reg_weight_ll,
      reg_weight_lu=FLAGS.reg_weight_lu,
      reg_weight_uu=FLAGS.reg_weight_uu,
      num_pairs_reg=FLAGS.num_pairs_reg,
      penalize_neg_agr=FLAGS.penalize_neg_agr,
      use_l2_cls=FLAGS.use_l2_cls,
      first_iter_original=FLAGS.first_iter_original,
      inductive=FLAGS.inductive,
      seed=FLAGS.seed,
      eval_acc_pred_by_agr=FLAGS.eval_acc_pred_by_agr,
      num_neighbors_pred_by_agr=FLAGS.num_neighbors_pred_by_agr,
      lr_decay_rate_cls=FLAGS.lr_decay_rate_cls,
      lr_decay_steps_cls=FLAGS.lr_decay_steps_cls,
      lr_decay_rate_agr=FLAGS.lr_decay_rate_agr,
      lr_decay_steps_agr=FLAGS.lr_decay_steps_agr,
      load_from_checkpoint=FLAGS.load_from_checkpoint)

  trainer.train(data)

if __name__ == '__main__':
  app.run(main)
