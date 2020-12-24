# Copyright 2020 Google LLC
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
"""Trains a GNN."""
import time

from absl import app
from absl import flags
import tensorflow as tf

from utils import load_dataset, build_model, cal_acc  # pylint: disable=g-multiple-import

flags.DEFINE_enum('dataset', 'cora', ['cora', 'citeseer'],
                  'The input dataset. Avaliable dataset now: cora, citeseer')
flags.DEFINE_enum('model', 'gat', ['gcn', 'gat', 'gin'],
                  'GNN model. Available model now: gcn, gat')
flags.DEFINE_float('dropout_rate', 0.6, 'Dropout probability')
flags.DEFINE_integer('gpu', '-1', 'Gpu id, -1 means cpu only')
flags.DEFINE_float('lr', 1e-2, 'Initial learning rate')
flags.DEFINE_integer('epochs', 1000, 'Number of training epochs')
flags.DEFINE_integer('num_layers', 2, 'Number of gnn layers')
flags.DEFINE_integer('mlp_layers', 2, 'Number of mlp layers')
flags.DEFINE_list('hidden_dim', [8], 'Dimension of gnn hidden layers')
flags.DEFINE_enum('optimizer', 'adam', ['adam', 'sgd'],
                  'Optimizer for training')
flags.DEFINE_integer('num_heads', 8, 'Number of multi-head attentions')
flags.DEFINE_integer('seed', 1234, 'Random seed')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 regularization')
flags.DEFINE_string('save_dir', 'models/cora/gcn',
                    'Directory stores trained model')
flags.DEFINE_boolean('save_best_val', False,
                     'Whether to save best val acc epoch or last epoch')
flags.DEFINE_boolean('learn_eps', False,
                     'Whether to learn the epsilon weighting for the nodes')
flags.DEFINE_boolean('normalize_adj', False, 'Whether to normalize adj matrix')
flags.DEFINE_boolean('sparse_features', True, 'Whether to use sparse features')

FLAGS = flags.FLAGS


def train(model, adj, features, labels, idx_train, idx_val, idx_test):
  """Train gnn model."""
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  best_val_acc = 0.0

  if FLAGS.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.lr)

  inputs = (features, adj)
  for epoch in range(FLAGS.epochs):
    epoch_start_time = time.time()

    with tf.GradientTape() as tape:
      output = model(inputs, training=True)
      train_loss = loss_fn(labels[idx_train], output[idx_train])
      # L2 regularization
      for weight in model.trainable_weights:
        train_loss += FLAGS.weight_decay * tf.nn.l2_loss(weight)

    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_acc = cal_acc(labels[idx_train], output[idx_train])

    # Evaluate
    output = model(inputs, training=False)
    val_loss = loss_fn(labels[idx_val], output[idx_val])
    val_acc = cal_acc(labels[idx_val], output[idx_val])

    if FLAGS.save_best_val:
      if val_acc >= best_val_acc:
       best_val_acc = val_acc
       model.save(FLAGS.save_dir)

    print('[%03d/%03d] %.2f sec(s) Train Acc: %.3f Loss: %.6f | Val Acc: %.3f loss: %.6f' % \
         (epoch + 1, FLAGS.epochs, time.time()-epoch_start_time, \
          train_acc, train_loss, val_acc, val_loss))
  
  if FLAGS.save_best_val == False:
    model.save(FLAGS.save_dir)
  print('Start Predicting...')
  model = tf.keras.models.load_model(FLAGS.save_dir)
  output = model(inputs, training=False)
  test_acc = cal_acc(labels[idx_test], output[idx_test])
  print('***Test Accuracy: %.3f***' % test_acc)


def main(_):

  if FLAGS.gpu == -1:
    device = '/cpu:0'
  else:
    device = '/gpu:{}'.format(FLAGS.gpu)

  with tf.device(device):
    tf.random.set_seed(FLAGS.seed)
    # Load the dataset and process features and adj matrix
    print('Loading {} dataset...'.format(FLAGS.dataset))
    adj, features, labels, idx_train, idx_val, idx_test = load_dataset(
        FLAGS.dataset, FLAGS.sparse_features, FLAGS.normalize_adj)
    num_classes = max(labels) + 1
    print('Build model...')
    model = build_model(FLAGS.model, FLAGS.num_layers, FLAGS.mlp_layers,
                        FLAGS.hidden_dim, num_classes, FLAGS.dropout_rate,
                        FLAGS.num_heads, FLAGS.learn_eps, FLAGS.sparse_features)
    print('Start Training...')
    train(model, adj, features, labels, idx_train, idx_val, idx_test)


if __name__ == '__main__':
  app.run(main)
