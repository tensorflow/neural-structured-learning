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
"""Default configuration parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

CONFIG = {
    'string': {
        'dataset': ('Dataset', 'WN18RR'),
        'model': ('Model', 'RotE'),
        'data_dir': ('Path to data directory', 'data/'),
        'save_dir': ('Path to logs directory', 'logs/'),
        'loss_fn': ('Loss function to use', 'SigmoidCrossEntropy'),
        'initializer': ('Which initializer to use', 'GlorotNormal'),
        'regularizer': ('Regularizer', 'N3'),
        'optimizer': ('Optimizer', 'Adam'),
        'bias': ('Bias term', 'learn'),
        'dtype': ('Precision to use', 'float32'),
    },
    'float': {
        'lr': ('Learning rate', 1e-3),
        'lr_decay': ('Learning rate decay', 0.96),
        'min_lr': ('Minimum learning rate decay', 1e-5),
        'gamma': ('Margin for distance-based losses', 0),
        'entity_reg': ('Regularization weight for entity embeddings', 0),
        'rel_reg': ('Regularization weight for relation embeddings', 0),
    },
    'integer': {
        'patience': ('Number of validation steps before early stopping', 20),
        'valid': ('Number of epochs before computing validation metrics', 5),
        'checkpoint': ('Number of epochs before checkpointing the model', 5),
        'max_epochs': ('Maximum number of epochs to train for', 400),
        'rank': ('Embeddings dimension', 500),
        'batch_size': ('Batch size', 500),
        'neg_sample_size':
            ('Negative sample size, -1 to use loss without negative sampling',
             50),
    },
    'boolean': {
        'train_c': ('Whether to train the hyperbolic curvature or not', True),
        'debug': ('If debug is true, only use 1000 examples for'
                  ' debugging purposes', False),
        'save_logs':
            ('Whether to save the training logs or print to stdout', True),
        'save_model': ('Whether to save the model weights', False)
    }
}
