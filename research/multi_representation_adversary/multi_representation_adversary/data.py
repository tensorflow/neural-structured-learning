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

"""Functions for processing datasets."""

import gin
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def convert_to_tuples(features):
  """Convert feature dictionary to (image, label) tuples."""
  return features["image"], features["label"]


@gin.configurable
def preprocess_image(image, label, height=32, width=32, num_channels=3):
  """Performs data augmentation of the image."""
  image = tf.image.resize_with_crop_or_pad(image, height + 8, width + 8)
  image = tf.image.random_crop(image, [height, width, num_channels])
  image = tf.image.random_flip_left_right(image)
  return image, label


def normalize_image(image, label):
  """Normalizes the image to within [0,1]."""
  image = tf.cast(image, dtype=tf.float32) / 255.0
  return image, label


@gin.configurable
def get_training_dataset(batch_size=128, dataset="cifar10", split="90%",
                         shuffle_buffer_size=45000):
  """Returns a tf.data.Dataset with preprocessed and batched training data."""
  ds = tfds.load(dataset, split=f"train[:{split}]")
  ds = ds.map(convert_to_tuples)
  ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True).repeat()
  return ds.map(preprocess_image).map(normalize_image).batch(batch_size)


@gin.configurable
def get_validation_dataset(batch_size=128, dataset="cifar10", split="10%"):
  """Returns a tf.data.Dataset with preprocessed and batched validation data."""
  ds = tfds.load(dataset, split=f"train[-{split}:]")
  return ds.map(convert_to_tuples).map(normalize_image).batch(batch_size)


@gin.configurable
def get_test_dataset(batch_size=128, dataset="cifar10"):
  """Returns a tf.data.Dataset with preprocessed and batched test data."""
  ds = tfds.load(dataset, split="test")
  return ds.map(convert_to_tuples).map(normalize_image).batch(batch_size)
