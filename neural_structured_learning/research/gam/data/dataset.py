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
import collections
import logging
import os
import pickle
import scipy

import numpy as np
import tensorflow as tf

from gam.data.preprocessing import split_train_val


class Dataset(object):
  """A container for datasets.

  In this dataset, each sample has the same number of features.
  This class manages different splits of the data for train, validation, test
  and unlabeled. These sets of samples are disjoint.
  """

  def __init__(self, name, features, labels, indices_train, indices_test,
               indices_val, indices_unlabeled, num_classes=None,
               feature_preproc_fn=lambda x: x):
    self.name = name
    self.features = features
    self.labels = labels

    self.indices_train = indices_train
    self.indices_val = indices_val
    self.indices_test = indices_test
    self.indices_unlabeled = indices_unlabeled
    self.feature_preproc_fn = feature_preproc_fn

    self.num_val = self.indices_val.shape[0]
    self.num_test = self.indices_test.shape[0]

    self.num_samples = labels.shape[0]
    self.features_shape = features.shape[1:]
    self.num_features = np.prod(features.shape[1:])
    self.num_classes = 1 + max(labels) if num_classes is None else num_classes

  @staticmethod
  def build_from_splits(name, inputs_train, labels_train, inputs_val,
                        labels_val, inputs_test, labels_test, inputs_unlabeled,
                        labels_unlabeled=None, num_classes=None,
                        feature_preproc_fn=lambda x: x):
    num_train = inputs_train.shape[0]
    num_val = inputs_val.shape[0]
    num_unlabeled = inputs_unlabeled.shape[0]
    num_test = inputs_test.shape[0]

    if labels_unlabeled is None:
      labels_unlabeled = np.zeros(shape=(num_unlabeled,),
                                  dtype=labels_train[0].dtype)
    features = np.concatenate(
      (inputs_train, inputs_val, inputs_unlabeled, inputs_test))
    labels = np.concatenate(
      (labels_train, labels_val, labels_unlabeled, labels_test))

    indices_train = np.arange(num_train)
    indices_val = np.arange(num_train, num_train+num_val)
    indices_unlabeled = np.arange(num_train+num_val,
                                  num_train+num_val+num_unlabeled)
    indices_test = np.arange(num_train+num_val+num_unlabeled,
                             num_train+num_val+num_unlabeled+num_test)

    return Dataset(name=name,
                   features=features,
                   labels=labels,
                   indices_train=indices_train,
                   indices_test=indices_test,
                   indices_val=indices_val,
                   indices_unlabeled=indices_unlabeled,
                   num_classes=num_classes,
                   feature_preproc_fn=feature_preproc_fn)

  @staticmethod
  def build_from_features(name, features, labels, indices_train, indices_test,
                          indices_val=None, indices_unlabeled=None,
                          percent_val=0.2, seed=None, num_classes=None,
                          feature_preproc_fn=lambda x: x):
    if indices_val is None:
      rng = np.random.RandomState(seed=seed)
      indices_train, indices_val = split_train_val(
        np.arange(indices_train.shape[0]), percent_val, rng)

    return Dataset(name=name,
                   features=features,
                   labels=labels,
                   indices_train=indices_train,
                   indices_test=indices_test,
                   indices_val=indices_val,
                   indices_unlabeled=indices_unlabeled,
                   num_classes=num_classes,
                   feature_preproc_fn=feature_preproc_fn)


  def copy(self, name=None, features=None, labels=None, indices_train=None,
           indices_test=None, indices_val=None, indices_unlabeled=None,
           num_classes=None, feature_preproc_fn=None):
    name = name if name is not None else self.name
    features = features if features is not None else self.features
    labels = labels if labels is not None else self.labels
    indices_train = (indices_train if indices_train is not None else
                     self.indices_train)
    indices_test = (indices_test if indices_test is not None else
                    self.indices_test)
    indices_val = indices_val if indices_val is not None else self.indices_val
    indices_unlabeled = (indices_unlabeled if indices_unlabeled is not None else
                         self.indices_unlabeled)
    num_classes = num_classes if num_classes is not None else self.num_classes
    feature_preproc_fn = (feature_preproc_fn if feature_preproc_fn is not None
                          else self.feature_preproc_fn)
    return Dataset(
      name=name,
      features=features,
      labels=labels,
      indices_train=indices_train,
      indices_test=indices_test,
      indices_val=indices_val,
      indices_unlabeled=indices_unlabeled,
      num_classes=num_classes,
      feature_preproc_fn=feature_preproc_fn)

  def copy_labels(self):
    return np.copy(self.labels)

  def get_features(self, indices):
    """Returns the features of the samples with the provided indices."""
    f = self.features[indices]
    f = self.feature_preproc_fn(f)
    return f

  def get_labels(self, indices):
    return self.labels[indices]

  def num_train(self):
    return self.indices_train.shape[0]

  def num_val(self):
    return self.indices_val.shape[0]

  def num_test(self):
    return self.indices_test.shape[0]

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
    # Remove the recently labeled samples from the unlabeled set.
    indices_samples = set(indices_samples)
    self.indices_unlabeled = np.asarray(
        [u for u in self.indices_unlabeled if u not in indices_samples])

  def update_labels(self, indices, new_labels):
    """Updates the labels of the samples with the provided indices.

    Arguments:
      indices: A list of integers representing sample indices.
      new_labels: A list of integers representing the new labels of th samples
        in indices_samples.
    """
    indices = np.asarray(indices)
    new_labels = np.asarray(new_labels)
    self.labels[indices] = new_labels

  def save_to_pickle(self, file_path):
    pickle.dump(self, open(file_path, 'w'))

  @staticmethod
  def load_from_pickle(file_path):
    dataset = pickle.load(open(file_path, 'r'))
    return dataset


class GraphDataset(Dataset):
  """Data container for SSL datasets."""
  class Edge(object):
    def __init__(self, src, tgt, weight=None):
      self.src = src
      self.tgt = tgt
      self.weight = weight

  def __init__(self, name, features, labels, edges, indices_train, indices_test,
               indices_val=None, indices_unlabeled=None, percent_val=0.2,
               seed=None, num_classes=None, feature_preproc_fn=lambda x: x):
    self.edges = edges

    if indices_val is None:
      rng = np.random.RandomState(seed=seed)
      indices_train, indices_val = split_train_val(
        np.arange(indices_train.shape[0]), percent_val, rng)

    super().__init__(
      name=name,
      features=features,
      labels=labels,
      indices_train=indices_train,
      indices_test=indices_test,
      indices_val=indices_val,
      indices_unlabeled=indices_unlabeled,
      num_classes=num_classes,
      feature_preproc_fn=feature_preproc_fn)

  def copy(self, name=None, features=None, labels=None, edges=None,
           indices_train=None, indices_test=None, indices_val=None,
           indices_unlabeled=None, num_classes=None, feature_preproc_fn=None):
    name = name if name is not None else self.name
    features = features if features is not None else self.features
    labels = labels if labels is not None else self.labels
    indices_train = (indices_train if indices_train is not None else
                     self.indices_train)
    indices_test = (indices_test if indices_test is not None else
                    self.indices_test)
    indices_val = indices_val if indices_val is not None else self.indices_val
    indices_unlabeled = (indices_unlabeled if indices_unlabeled is not None else
                         self.indices_unlabeled)
    num_classes = num_classes if num_classes is not None else self.num_classes
    feature_preproc_fn = (feature_preproc_fn if feature_preproc_fn is not None
                          else self.feature_preproc_fn)
    edges = edges if edges is not None else self.edges
    return GraphDataset(
      name=name,
      features=features,
      labels=labels,
      edges=edges,
      indices_train=indices_train,
      indices_test=indices_test,
      indices_val=indices_val,
      indices_unlabeled=indices_unlabeled,
      num_classes=num_classes,
      feature_preproc_fn=feature_preproc_fn)

  def get_edges(self, src_labeled=None, tgt_labeled=None,
                label_must_match=False):
    labeled_mask = np.full((self.num_samples,), False)
    labeled_mask[self.get_indices_train()] = True

    def _labeled_cond(idx, is_labeled):
      return (is_labeled is None) or (is_labeled == labeled_mask[idx])

    def _agreement_cond(edge):
      return self.get_labels(edge.src) == self.get_labels(edge.tgt)

    agreement_cond = _agreement_cond if label_must_match else lambda e: True

    return [e for e in self.edges
            if _labeled_cond(e.src, src_labeled) and \
            _labeled_cond(e.tgt, tgt_labeled) and \
            agreement_cond(e)]


class PlanetoidDataset(GraphDataset):
  """Data container for Planetoid datasets."""

  def __init__(self, name, adj, features, train_mask, val_mask, test_mask,
               labels, row_normalize=False):

    # Extract train, val, test, unlabeled indices.
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    val_indices = np.where(val_mask)[0]
    unlabeled_mask = np.logical_not(train_mask | test_mask | val_mask)
    unlabeled_indices = np.where(unlabeled_mask)[0]

    # Extract node features.
    if row_normalize:
      features = self.preprocess_features(features)

    features = np.float32(features.todense())

    # Extract labels.
    labels = np.argmax(labels, axis=-1)
    num_classes = max(labels) + 1

    # Extract edges.
    adj = scipy.sparse.coo_matrix(adj)
    edges = [self.Edge(src, tgt, val)
             for src, tgt, val in zip(adj.row, adj.col, adj.data)]

    # Convert to Dataset format.
    super().__init__(
      name=name,
      features=features,
      labels=labels,
      edges=edges,
      indices_train=train_indices,
      indices_test=test_indices,
      indices_val=val_indices,
      indices_unlabeled=unlabeled_indices,
      num_classes=num_classes,
      feature_preproc_fn=lambda x: x)

  @staticmethod
  def preprocess_features(features):
    """Row-normalize feature matrix."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


class CotrainDataset(object):
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

  def get_edges(self, src_labeled=None, tgt_labeled=None,
                label_must_match=False):
    return self.dataset.get_edges(
      src_labeled=src_labeled,
      tgt_labeled=tgt_labeled,
      label_must_match=label_must_match)