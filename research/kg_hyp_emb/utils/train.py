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
"""Training utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from kg_hyp_emb.config import CONFIG
import numpy as np

FLAGS = flags.FLAGS


def get_config_dict():
  """Maps FLAGS to dictionnary in order to save it in json format."""
  config = {}
  for _, arg_dict in CONFIG.items():
    for arg, _ in arg_dict.items():
      config[arg] = getattr(FLAGS, arg)
  return config


def count_params(model):
  """Counts the total number of trainable parameters in a KG embedding model.

  Args:
    model: A tf.keras.Model KG embedding model.

  Returns:
    Integer representing the number of trainable variables.
  """
  total = 0
  for x in model.trainable_variables:
    total += np.prod(x.shape)
  return total


def avg_both(mrs, mrrs, hits):
  """Aggregate metrics for left- and right-hand-side predictions.

  Args:
    mrs: Dictionary with mean ranks for lhs and rhs.
    mrrs: Dictionary with mean reciprocical ranks for lhs and rhs.
    hits: Dictionary with hits at 1, 3, 10 for lhs and rhs.

  Returns:
    Dictionary with averaged metrics.
  """
  mr = (mrs['lhs'] + mrs['rhs']) / 2.
  mrr = (mrrs['lhs'] + mrrs['rhs']) / 2.
  h = []
  for k in [1, 3, 10]:
    h += [(hits['lhs'][k] + hits['rhs'][k]) / 2.]
  return {'MR': mr, 'MRR': mrr, 'hits@[1,3,10]': h}


def format_metrics(metrics, split):
  """Formats metrics for logging.

  Args:
    metrics: Dictionary with metrics.
    split: String indicating the KG dataset split.

  Returns:
    String with formatted metrics.
  """
  result = '\t {} MR: {:.2f} | '.format(split, metrics['MR'])
  result += 'MRR: {:.3f} | '.format(metrics['MRR'])
  result += 'H@1: {:.3f} | '.format(metrics['hits@[1,3,10]'][0])
  result += 'H@3: {:.3f} | '.format(metrics['hits@[1,3,10]'][1])
  result += 'H@10: {:.3f}'.format(metrics['hits@[1,3,10]'][2])
  return result
