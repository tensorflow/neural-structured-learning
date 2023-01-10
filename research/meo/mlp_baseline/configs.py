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

"""Experiment configurations such as model architecture and dataset configurations.

This file contains wrappers for useful configurations for the dataset and the
architecture used in the experiment. The intent is to have these configurations
easily available.
"""

from meo.mlp_baseline import data_utils

OBFUSCATION_CAPTIONS = [
    'A clean image.',
    'An image under color line shift.',
    'A composed image.',
    'An image with halftoning.',
    'An image with an icon overlaid.',
    'An image with an image overlaid.',
    'An interleaved image.',
    'An image with inverted colors.',
    'An image under line shift.',
    'An image under a perspective transform.',
    'An image with rotated blocks.',
    'An image with rotated hue.',
    'A rotated image.',
    'An image with style transfer.',
    'An image with swirl warp.',
    'An image with text overlaid.',
    'A texturized image.',
    'A perturbed image.'
]


class ModelConfig(object):
  r"""Wrapper for architecture configurations.

  This class provides access to configurations for the architecture used in the
  experiment.

  Attributes
    embed_dim: Dimension of the feature representation of the architecture.
    input_dim: Dimension of the input of the architecture.
    model_link: Link to the model in tfhub.
  """

  def __init__(self, model_type: str):
    self.input_dim = (224, 224, 3)
    if model_type == 'resnet50':
      self.embed_dim = 2048
      self.model_link = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5'
    elif model_type == 'resnet101':
      self.embed_dim = 2048
      self.model_link = 'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5'
    elif model_type == 'resnet152':
      self.embed_dim = 2048
      self.model_link = 'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5'
    elif model_type == 'bit-m152x4':
      self.embed_dim = 8192
      self.model_link = 'https://tfhub.dev/google/bit/m-r152x4/1'
    elif model_type == 'vitb8_clf':
      self.embed_dim = None  # Classifier
      self.model_link = 'https://tfhub.dev/sayakpaul/vit_b8_classification/1'
    else:
      raise ValueError('Model type not understood: {}'.format(model_type))


class DatasetConfig(object):
  r"""Wrapper for dataset configurations.

  This class provides access to configurations for the dataset used in the
  experiment.

  TODO(smyrnisg): Add support for datasets other than ImageNet.

  Attributes
    num_classes: Number of classes in the Dataset
    train_size: Number of samples used for training.
    eval_size: Number of samples used for evaluation.
    dataset_name_compat: For compatibility with the name of predefined dataset.
  """

  def __init__(self, dataset: str):
    if dataset == 'imagenet':
      self.dataset_name_compat = data_utils.Dataset.IMAGENET
      self.num_classes = 1000
      self.train_size = 1281167
      self.eval_size = 50000

    else:
      raise ValueError('Dataset not understood: {}'.format(dataset))
