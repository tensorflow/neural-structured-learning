import os
import time
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

from utils import load_dataset, build_model, cal_acc

flags.DEFINE_string('dataset', 'cora', 
                    'The input dataset. Avaliable dataset now: cora')
flags.DEFINE_string('model', 'gcn',
                    'GNN model. Available model now: gcn')
flags.DEFINE_float('dropout', 0.5, 'Dropout probability')
flags.DEFINE_integer('gpu', '-1', 'Gpu id, -1 means cpu only')
flags.DEFINE_float('lr', 1e-2, 'Initial learning rate')
flags.DEFINE_integer('epochs', 200, 'Number of training epochs')
flags.DEFINE_integer('num_layers', 2, 'Number of gnn hidden layers')
flags.DEFINE_list('hidden_dim', [32], 'Dimension of gnn hidden layers')
flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 regularization')

FLAGS = flags.FLAGS


def train(model, adj, features, y_train, y_val):
    """Train gnn model."""
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_last_id = y_train.shape.as_list()[0]
    val_last_id = y_val.shape.as_list()[0] + train_last_id

    if FLAGS.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.lr)
    
    
    for epoch in range(FLAGS.epochs):
        epoch_start_time = time.time()
        
        with tf.GradientTape() as tape:
            output = model(features, adj)
            train_loss = loss_fn(y_train, output[:train_last_id])
        gradients = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_acc = cal_acc(y_train, output[:train_last_id])
        
        # Evaluate
        output = model(features, adj, training=False)
        val_loss = loss_fn(y_val, output[train_last_id:val_last_id])
        val_acc = cal_acc(y_val, output[train_last_id:val_last_id])
        
        print('[%03d/%03d] %.2f sec(s) Train Acc: %.3f Loss: %.6f | Val Acc: %.3f loss: %.6f' % \
             (epoch + 1, FLAGS.epochs, time.time()-epoch_start_time, \
              train_acc, train_loss, val_acc, val_loss))



def main(_):
    
    if FLAGS.gpu == -1:
        device = "/cpu:0"
    else:
        device = "/gpu:{}".format(FLAGS.gpu)
    
    with tf.device(device):
        tf.random.set_seed(1234)
        # Load the dataset and process features and adj matrix
        print('Loading {} dataset...'.format(FLAGS.dataset))
        adj, features, y_train, y_val, y_test = load_dataset(FLAGS.dataset)
        features_dim = features.shape[1]
        num_classes = max(y_test) + 1
        print('Build model...')
        model = build_model(FLAGS.model, features_dim, FLAGS.num_layers, 
                            FLAGS.hidden_dim, num_classes, FLAGS.dropout)
    
        print('Start Training...')
        train(model, adj, features, y_train, y_val)
    

if __name__ == '__main__':
    app.run(main)

