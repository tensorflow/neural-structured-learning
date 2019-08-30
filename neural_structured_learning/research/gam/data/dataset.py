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
"""Data containers for Graph Agreement Models."""
import abc
import collections
import logging
import os
import pickle

import numpy as np
import tensorflow as tf


class Dataset(object):
  """Interface for different types of datasets."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def num_train(self):
    pass

  @abc.abstractmethod
  def num_val(self):
    pass

  @abc.abstractmethod
  def num_test(self):
    pass

  @abc.abstractmethod
  def num_unlabeled(self):
    pass

  @abc.abstractmethod
  def copy_labels(self):
    pass

  def save_to_pickle(self, file_path):
    pickle.dump(self, open(file_path, 'w'))

  @staticmethod
  def load_from_pickle(file_path):
    dataset = pickle.load(open(file_path, 'r'))
    return dataset


class FixedDataset(Dataset):
  """A dataset containing features of fixed size.

  In this dataset, each sample has the same number of features.
  This class manages different splits of the data for train, validation, test
  and unlabeled. These sets of samples are disjoint.
  """

  def __init__(self,
               x_train,
               y_train,
               x_val,
               y_val,
               x_test,
               y_test,
               x_unlabeled,
               y_unlabeled=None,
               num_classes=None,
               feature_preproc_fn=lambda x: x):
    n_train = x_train.shape[0]
    n_val = x_val.shape[0]
    n_test = x_test.shape[0]
    n_unlabeled = x_unlabeled.shape[0]

    if y_unlabeled is None:
      y_unlabeled = np.zeros(shape=(n_unlabeled,), dtype=y_train.dtype)

    # Concatenate samples.
    self.features = np.concatenate((x_train, x_val, x_unlabeled, x_test))
    self.labels = np.concatenate((y_train, y_val, y_unlabeled, y_test))

    self._num_features = np.prod(self.features.shape[1:])
    self._num_classes = 1 + max(self.labels) if num_classes is None else \
                        num_classes
    self._num_samples = n_train + n_val + n_unlabeled + n_test

    self.indices_train = np.arange(n_train)
    self.indices_val = np.arange(n_train, n_train+n_val)
    self.indices_unlabeled = np.arange(n_train+n_val, n_train+n_val+n_unlabeled)
    self.indices_test = np.arange(n_train+n_val+n_unlabeled, self._num_samples)

    self.feature_preproc_fn = feature_preproc_fn

  def copy_labels(self):
    return np.copy(self.labels)

  def update_labels(self, indices_samples, new_labels):
    """Updates the labels of the samples with the provided indices.

    Arguments:
      indices_samples: A list of integers representing sample indices.
      new_labels: A list of integers representing the new labels of th samples
        in indices_samples.
    """
    indices_samples = np.asarray(indices_samples)
    new_labels = np.asarray(new_labels)
    self.labels[indices_samples] = new_labels

  def get_features(self, indices):
    """Returns the features of the samples with the provided indices."""
    f = self.features[indices]
    f = self.feature_preproc_fn(f)
    return f

  def get_labels(self, indices):
    return self.labels[indices]

  @property
  def num_features(self):
    return self._num_features

  @property
  def features_shape(self):
    """Returns the shape of the input features, not including batch size."""
    return self.features.shape[1:]

  @property
  def num_classes(self):
    return self._num_classes

  @property
  def num_samples(self):
    return self._num_samples

  def num_train(self):
    return self.indices_train.shape[0]

  def num_val(self):
    return self.indices_val.shape[0]

  @property
  def num_test(self):
    return self.indices_test.shape[0]

  @property
  def num_unlabeled(self):
    return self.indices_unlabeled.shape[0]

  def get_indices_train(self):
    return self.indices_train

  def get_indices_unlabeled(self):
    return self.indices_unlabeled

  def get_indices_val(self):
    return self.indices_val

  def get_indices_test(self):
    return self.indices_test

  def label_samples(self, indices_samples, new_labels):
    """Updates the labels of the samples with the provided indices.

    Arguments:
      indices_samples: Array of integers representing the sample indices to
        update. These must samples that are not already considered training
        samples, otherwise they will be duplicated.
      new_labels: Array of integers containing the new labels for each of the
        samples in indices_samples.
    """
    # Update the labels.
    self.update_labels(indices_samples, new_labels)

    # Mark the newly labeled nodes as training samples. Note that for
    # efficiency we simply concatenate the new samples to the existing training
    # indices, without checking if they already exist.
    self.indices_train = np.concatenate((self.indices_train, indices_samples),
                                        axis=0)
    #  Remove the recently labeled samples from the unlabeled set.
    indices_samples = set(indices_samples)
    self.indices_unlabeled = np.asarray(
        [u for u in self.indices_unlabeled if u not in indices_samples])


class CotrainDataset(Dataset):
  """A wrapper around a Dataset object, adding co-training functionality.

  Attributes:
    dataset: A Dataset object.
    keep_label_proportions: A boolean specifying whether every self-labeling
      step will keep the label proportions from the original training data.
      If so, this class keeps statistics of the label distribution.
    inductive: A boolean specifying if the dataset is inductive. If True,
      then the unlabeled samples include the validation and test samples.
    labels_original: Stores the original labels of all samples in `dataset`.
      We do this because via the self-labeling step, the method
      `label_samples` can rewrite the original (correct) labels, which are
      needed for evaluation.
    label_prop: Stores the ratio of samples for each label in the original
      training set.
  """

  def __init__(self, dataset, keep_label_proportions=False, inductive=False):
    self.dataset = dataset
    self.keep_label_proportions = keep_label_proportions
    self.inductive = inductive

    # Make a copy of the labels because we will modify them through
    # self-labeling. We need the originals to monitor the accuracy.
    self.labels_original = dataset.copy_labels()

    # If we are in an inductive setting, the validation and test samples are
    # not seen at training time, so they cannot be labeled via self-labeling.
    # Otherwise, they are treated as any unlabeled samples.
    if not self.inductive:
      self.dataset.indices_unlabeled = np.concatenate(
          (dataset.indices_unlabeled,
           dataset.indices_val,
           dataset.indices_test))

    # If when labeling new data we want to keep the proportions of the labels
    # in the original labeled data, then save these proportions.
    if keep_label_proportions:
      labeled_indices = list(self.get_indices_train())
      labeled_indices_labels = dataset.get_labels(labeled_indices)
      label_counts = collections.Counter(labeled_indices_labels)
      num_labels = np.float(len(labeled_indices))
      assert num_labels > 0, 'There are no labeled samples in the dataset.'
      self.label_prop = {
          label: count / num_labels for label, count in label_counts.items()}
    else:
      self.label_prop = None

  def label_samples(self, indices_samples, new_labels):
    """Updates the labels of the samples with the provided indices.

    Arguments:
      indices_samples: Array of integers representing the sample indices to
        update. These must samples that are not already considered training
        samples, otherwise they will be duplicated.
      new_labels: Array of integers containing the new labels for each of the
        samples in indices_samples.
    """
    self.dataset.label_samples(indices_samples, new_labels)

  def _compute_label_correctness(self, indices):
    if indices.shape[0] == 0:
      return 0.0
    labels = self.get_labels(indices)
    labels_orig = self.get_original_labels(indices)
    correct_labels = np.sum(labels == labels_orig)
    ratio_correct = float(correct_labels) / indices.shape[0]
    return ratio_correct

  def compute_dataset_statistics(self, selected_samples, summary_writer, step):
    """Computes statistics about the correctness of the labels of a Dataset."""
    # Compute how many new nodes are labeled correctly.
    ratio_correct_new_labels = self._compute_label_correctness(selected_samples)

    # Compute how many nodes are labeled correctly in total.
    ratio_correct_total = self._compute_label_correctness(
        self.get_indices_train())

    num_new_labels = len(selected_samples)
    logging.info('Number of newly labeled nodes: %d', num_new_labels)
    logging.info('Ratio correct new labels: %f. '
                 'Ratio correct labels total: %f.',
                 ratio_correct_new_labels, ratio_correct_total)

    # Compute how many test nodes have been labeled, and their correctness.
    # Note: This is an expensive step, and should be skipped for large datasets.
    if not self.inductive:
      indices_train = set(self.get_indices_train())
      indices_test = set(self.get_indices_test())
      labeled_test_indices = indices_train.intersection(indices_test)
      labeled_test_indices = np.asarray(list(labeled_test_indices))
      ratio_correct_labeled_test = self._compute_label_correctness(
          labeled_test_indices)
      logging.info('Number of labeled test nodes: %d',
                   labeled_test_indices.shape[0])
      logging.info('Ratio correct labels for labeled test nodes: %f.',
                   ratio_correct_labeled_test)

    # Compute the distribution of labels for all labeled nodes.
    labels_new_labeled_nodes = self.get_labels(selected_samples)
    labels_counts = collections.Counter(labels_new_labeled_nodes)
    logging.info('--------- Label distribution for NEW nodes -----------------')
    logging.info(labels_counts)

    labels_all_labeled_nodes = self.get_labels(self.get_indices_train())
    labels_counts = collections.Counter(labels_all_labeled_nodes)
    logging.info('-------- Label distribution for ALL LABELED nodes ----------')
    logging.info(labels_counts)

    # Show Tensorboard summaries.
    if summary_writer is not None:
      summary = tf.Summary()
      summary.value.add(
          tag='data/ratio_correct_labels_new',
          simple_value=ratio_correct_new_labels)
      summary.value.add(
          tag='data/ratio_correct_labels_total',
          simple_value=ratio_correct_total)
      summary.value.add(tag='data/num_new_nodes', simple_value=num_new_labels)
      if not self.inductive:
        summary.value.add(
            tag='data/num_labeled_test',
            simple_value=labeled_test_indices.shape[0])
        if labeled_test_indices.shape[0] > 0:
          summary.value.add(
              tag='data/ratio_correct_labeled_test',
              simple_value=ratio_correct_labeled_test)
      summary_writer.add_summary(summary, step)
      summary_writer.flush()

    return ratio_correct_new_labels, ratio_correct_total

  def get_indices_train(self):
    """Returns the indices of the labeled samples that can be used to train."""
    return self.dataset.get_indices_train()

  def get_indices_val(self):
    """Returns the indices of the samples that can be used to validate."""
    return self.dataset.get_indices_val()

  def get_indices_test(self):
    """Returns the indices of the samples that can be used to test."""
    return self.dataset.get_indices_test()

  def get_indices_unlabeled(self):
    """Returns the indices of the samples that can be self-labeled."""
    return self.dataset.get_indices_unlabeled()

  def get_features(self, indices):
    """Returns the features of the requested indices."""
    return self.dataset.get_features(indices)

  def get_labels(self, indices):
    """Returns the labels of the requested indices."""
    return self.dataset.get_labels(indices)

  def get_original_labels(self, indices):
    """Returns the labels of the requested indices in the original dataset.

    This is different than `get_labels` because the current labels of a sample
    may have been changed via self-labeling.

    Args:
      indices: A list or array of sample indices.
    Returns:
      An array of labels.
    """
    return self.labels_original[indices]

  @property
  def num_samples(self):
    """Returns the total number of samples in the dataset."""
    return self.dataset.num_samples

  @property
  def num_features(self):
    """Returns the number of features of the samples in the dataset."""
    return self.dataset.num_features

  @property
  def features_shape(self):
    """Returns the shape of the input features, not including batch size."""
    return self.dataset.features_shape

  @property
  def num_classes(self):
    """Returns the number of classes of the samples in the dataset."""
    return self.dataset.num_classes

  def num_train(self):
    """Returns the number of training samples in the dataset."""
    return self.dataset.num_train()

  def num_val(self):
    """Returns the number of validation samples in the dataset."""
    return self.dataset.num_val()

  def num_test(self):
    """Returns the number of test samples in the dataset."""
    return self.dataset.num_test()

  def num_unlabeled(self):
    """Returns the number of unlabeled samples in the dataset."""
    return self.dataset.num_unlabeled()

  def save_state_to_file(self, output_path):
    """Saves the class attributes that are being updated during cotraining."""
    indices_unlabeled = self.get_indices_unlabeled()
    indices_train = self.get_indices_train()
    labels_train = self.get_labels(indices_train)
    with open(os.path.join(output_path, 'indices_train.txt'), 'w') as f:
      np.savetxt(f, indices_train[None], delimiter=',')
    with open(os.path.join(output_path, 'indices_unlabeled.txt'), 'w') as f:
      np.savetxt(f, indices_unlabeled[None], delimiter=',')
    with open(os.path.join(output_path, 'labels_train.txt'), 'w') as f:
      np.savetxt(f, labels_train[None], delimiter=',')

  def restore_state_from_file(self, path):
    """Restore the class attributes that are being updated during cotraining."""
    file_indices_train = os.path.join(path, 'indices_train.txt')
    file_indices_unlabeled = os.path.join(path, 'indices_unlabeled.txt')
    file_labels_train = os.path.join(path, 'labels_train.txt')
    indices_type = (np.uint32
                    if self.dataset.num_samples < np.iinfo(np.uint32).max
                    else np.uint64)
    if os.path.exists(file_indices_train) \
        and os.path.exists(file_indices_unlabeled) \
        and os.path.exists(file_labels_train):
      with open(file_indices_train, 'r') as f:
        indices_train = np.genfromtxt(f, delimiter=',', dtype=indices_type)
      with open(file_indices_unlabeled, 'r') as f:
        indices_unlabeled = np.genfromtxt(f, delimiter=',', dtype=indices_type)
      with open(file_labels_train, 'r') as f:
        labels_train = np.genfromtxt(f, delimiter=',', dtype=np.uint32)
      self.dataset.indices_train = indices_train
      self.dataset.indices_unlabeled = indices_unlabeled
      self.dataset.update_labels(indices_train, labels_train)
    else:
      logging.info('No data checkpoints found.')

  def copy_labels(self):
    return self.dataset.copy_labels()
