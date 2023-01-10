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

"""Parallel obfuscations dataset with TPU support.

This file provides a wrapper for a set of obfuscations with TPU support.

The main classes contained in this file are the following:
- ObfuscatedImageDataset: For compatibility, this class assumes that the
  filename the dataset was saved with also contains the obfuscation. Internally,
  it is assumed that the dataset contains fields of the form
  'image_{obfuscation}'.
- ObfuscatedEmbeddingDatasetEmbedding: This class assumes that the dataset
  contains the embeddings of the various views of the image instead, under the
  'embed' field.
"""
import abc
from typing import Optional, Sequence, Tuple

from meo.mlp_baseline import data_utils
import tensorflow as tf
import tensorflow_datasets as tfds


class ObfuscatedDatasetBase(object, metaclass=abc.ABCMeta):
  """Base class for the obfuscation datasets.

  This class provides the functionality required to run experiments with TPU
  support, via the input_fn function. Subclasses should implement the
  _read_dataset method, in order to read the input data from its respective
  source.

  Attributes:
    data_dir: Directory from which to read the data.
    split: Name of the split used (one of 'train' or 'test').
    batch_size: Batch size for the data.
    is_train: Whether the dataset is used for training or not.
  """

  def __init__(
      self,
      data_dir: str,
      batch_size: int,
      split: str):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.split = split
    self.is_train = split.startswith('train')

  @abc.abstractmethod
  def _read_dataset(
      self,
      context: Optional[tf.distribute.InputContext]
  ) -> tf.data.Dataset:
    """Abstract method that reads a dataset from the appropriate source.

    Args:
      context: A context object (tf.distribute.InputContext). If provided, this
        contains information about the TPU setup.

    Returns:
      A tf.data.Dataset, containing the data of the required shard of the
      dataset. The dataset returned should not be batched.
    """

  def input_fn(
      self,
      context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Function for the generation of a dataset for use in TPU.

    This function shards and batches the provided datasets, and
    allows for execution of the code in TPU, using TPUStrategy.

    Args:
      context: A context object (tf.distribute.InputContext). If provided, this
        contains information about the TPU setup.

    Returns:
      A tf.data.Dataset, containing the shard of the obfuscated dataset that we
      want.
    """

    dataset = self._read_dataset(context)

    if self.is_train:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(1024)

    dataset = dataset.batch(
        self.batch_size, drop_remainder=self.is_train
    )

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if self.is_train:
      options = tf.data.Options()
      options.deterministic = False
      dataset = dataset.with_options(options)

    return dataset


class ObfuscatedImageDataset(ObfuscatedDatasetBase):
  """Class wrapping a dataset of obfuscated images.

  This class returns a dataset of tf.train.Examples, each containing the
  following fields:
  - 'image_{obf}': The obfuscated image, where obf is each of the valid
    obfuscations. The images have shapes (height, width, 3).
  - 'label': The label of the respective input samples.

  Attributes:
    dataset: An object from the data_utils.Dataset enumeration, identifying the
      dataset from which the data is derived.
    obfuscation_list: Obfuscations to be used in our dataset. Currently only
      supports a list with one element (one obfuscation).
  """

  def __init__(
      self,
      dataset: data_utils.Dataset,
      data_dir: str,
      obfuscation_list: Sequence[str],
      split: str,
      batch_size: int
    ):
    super().__init__(data_dir=data_dir, batch_size=batch_size, split=split)
    self.dataset = dataset
    self.obfuscation_list = obfuscation_list

  def _read_dataset(
      self,
      context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    obfuscation = self.obfuscation_list[0]
    split = data_utils.Split.from_string(self.split)
    base_subset = data_utils.get_subsets(split, [obfuscation])[0]

    input_pipeline_id = 0
    num_input_pipelines = 1
    if context:
      input_pipeline_id = context.input_pipeline_id
      num_input_pipelines = context.num_input_pipelines

    start, end = data_utils.shard(
        self.dataset,
        split,
        obfuscation=base_subset,
        shard_index=input_pipeline_id,
        num_shards=num_input_pipelines)

    base_subset = tfds.core.ReadInstruction(
        base_subset, from_=start, to=end, unit='abs')

    return_dataset = tfds.load(
        self.dataset.value, data_dir=self.data_dir, split=base_subset)
    return return_dataset


class ObfuscatedEmbeddingDataset(ObfuscatedDatasetBase):
  """"Class to read a dataset of embeddings of obfuscated images.

  This class is used to read a tfrecord which contains the embeddings of the
  images directly. The dataset needs to contain two fields:
  - 'embed': The embeddings of the images. This field should contain a float
      vector of dimension (num_views * embed_dim), so it can then be reshaped to
      a 2D tensor.
  - 'label': The int label of the corresponding sample.

  Attributes:
    embed_dim: The dimension of the embeddings.
    num_views: Number of different views of each image to be found in the
      dataset (equal to the number of obfuscations + 1 for the clean image).
  """

  def __init__(
      self,
      data_dir: str,
      embed_dim: int,
      num_views: int,
      batch_size: int,
      split: str
  ):
    super().__init__(data_dir=data_dir, batch_size=batch_size, split=split)
    # Example filename: data_dir/<dataset name>_<split>.<suffix>
    self._filenames = sorted(
        tf.io.gfile.glob(self.data_dir + '/*{}*'.format(split)))
    self.embed_dim = embed_dim
    self.num_views = num_views

  def _parse_fn(self, tf_examples: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Method to parse a batch of tf.train.Example protos for this dataset.

    Args:
      tf_examples: The batch of tf.train.Example protos containing the
      information for the dataset.

    Returns:
      A tuple of (features, labels). The tensor 'features' has shape
        (batch_size, num_views, embed_dim).
    """
    keys_to_features = {
        'embed':
            tf.io.FixedLenFeature((self.num_views * self.embed_dim),
                                  tf.float32),
        'label':
            tf.io.FixedLenFeature((), tf.int64),
    }
    parsed = tf.io.parse_example(tf_examples, keys_to_features)
    label = parsed['label']
    features = tf.reshape(parsed['embed'], [-1, self.num_views, self.embed_dim])
    return features, label

  def _read_file(self, filename: str) -> tf.data.Dataset:
    dataset = tf.data.TFRecordDataset(filename)

    # The parsing function internally expects a batch of records, hence the
    # batching below. The actual batch size of this is irrelevant.
    dataset = dataset.batch(32)
    dataset = dataset.map(self._parse_fn)
    return dataset

  def _read_dataset(
      self,
      context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(self._filenames)
    if context and context.num_input_pipelines > 1:
      dataset = dataset.shard(
          context.num_input_pipelines,
          context.input_pipeline_id
      )
    dataset = dataset.interleave(
        self._read_file,
        deterministic=not self.is_train,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.unbatch()
    return dataset
