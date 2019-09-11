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
"""Trainer for classification models for Graph Agreement Models without a graph.

This class contains functionality that allows for training a classification
model to be used as part of Graph Agreement Models.
This implementation does not use a provided graph, but samples random pairs
of samples.
"""
import logging
import os

from gam.trainer.trainer_base import batch_iterator
from gam.trainer.trainer_base import Trainer

import numpy as np
import tensorflow as tf


class TrainerClassification(Trainer):
  """Trainer for the classifier component of a Graph Agreement Model.

  Attributes:
    model: A Model object that is used to provide the architecture of the
      classification model.
    is_train: A placeholder for a boolean value specyfing if the model is used
      for train or evaluation.
    data: A CotrainDataset object.
    trainer_agr: A TrainerArgeement or TrainerPerfectAgreement object.
    optimizer: Optimizer used for training the classification model.
    batch_size: Batch size for used when training and evaluating the
      classification model.
    gradient_clip: A float number representing the maximum gradient norm allowed
      if we do gradient clipping. If None, no gradient clipping is performed.
    min_num_iter: An integer representing the minimum number of iterations to
      train the classification model.
    max_num_iter: An integer representing the maximum number of iterations to
      train the classification model.
    num_iter_after_best_val: An integer representing the number of extra
      iterations to perform after improving the validation set accuracy.
    max_num_iter_cotrain: An integer representing the maximum number of cotrain
      iterations to train for.
    reg_weight_ll: A float representing the weight of the agreement loss term
      component of the classification model loss function, between
      labeled-labeled pairs of samples.
    reg_weight_lu: A float representing the weight of the agreement loss term
      component of the classification model loss function, between
      labeled-unlabeled pairs of samples.
    reg_weight_uu: A float representing the weight of the agreement loss term
      component of the classification model loss function, between
      unlabeled-unlabeled pairs of samples.
    num_pairs_reg: An integer representing the number of sample pairs of each
      type (LL, LU, UU) to include in each computation of the classification
      model loss.
    iter_cotrain: A Tensorflow variable containing the current cotrain
      iteration.
    enable_summaries: Boolean specifying whether to enable variable summaries.
    summary_step: Integer representing the summary step size.
    summary_dir: String representing the path to a directory where to save the
      variable summaries.
    logging_step: Integer representing the number of iterations after which
      we log the loss of the model.
    eval_step: Integer representing the number of iterations after which we
      evaluate the model.
    warm_start: Whether the model parameters are initialized at their
      best value in the previous cotrain iteration. If False, they are
      reinitialized.
    gradient_clip=None,
    abs_loss_chg_tol: A float representing the absolute tolerance for checking
      if the training loss has converged. If the difference between the current
      loss and previous loss is less than `abs_loss_chg_tol`, we count this
      iteration towards convergence (see `loss_chg_iter_below_tol`).
    rel_loss_chg_tol: A float representing the relative tolerance for checking
      if the training loss has converged. If the ratio between the current loss
      and previous loss is less than `rel_loss_chg_tol`, we count this iteration
      towards convergence (see `loss_chg_iter_below_tol`).
    loss_chg_iter_below_tol: An integer representing the number of consecutive
      iterations that pass the convergence criteria before stopping training.
    checkpoints_dir: Path to the folder where to store TensorFlow model
      checkpoints.
    weight_decay: Weight for the weight decay term in the classification
      model loss.
    weight_decay_schedule: Schedule how to adjust the classification weight
      decay weight after every cotrain iteration.
    penalize_neg_agr: Whether to not only encourage agreement between samples
      that the agreement model believes should have the same label, but also
      penalize agreement when two samples agree when the agreement model
      predicts they should disagree.
    use_l2_clssif: Whether to use L2 loss for classification, as opposed to the
      whichever loss is specified in the provided model_cls.
    first_iter_original:  A boolean specifying whether the first cotrain
      iteration trains the original classification model (with no agreement
      term).
    seed: Seed used by all the random number generators in this class.
  """

  def __init__(self,
               model,
               is_train,
               data,
               trainer_agr,
               optimizer,
               lr_initial,
               batch_size,
               min_num_iter,
               max_num_iter,
               num_iter_after_best_val,
               max_num_iter_cotrain,
               reg_weight_ll,
               reg_weight_lu,
               reg_weight_uu,
               num_pairs_reg,
               iter_cotrain,
               enable_summaries=False,
               summary_step=1,
               summary_dir=None,
               warm_start=False,
               gradient_clip=None,
               logging_step=1,
               eval_step=1,
               abs_loss_chg_tol=1e-10,
               rel_loss_chg_tol=1e-7,
               loss_chg_iter_below_tol=30,
               checkpoints_dir=None,
               weight_decay=None,
               weight_decay_schedule=None,
               penalize_neg_agr=False,
               first_iter_original=True,
               use_l2_classif=True,
               seed=None,
               lr_decay_steps=None,
               lr_decay_rate=None):
    super(TrainerClassification, self).__init__(
        model=model,
        abs_loss_chg_tol=abs_loss_chg_tol,
        rel_loss_chg_tol=rel_loss_chg_tol,
        loss_chg_iter_below_tol=loss_chg_iter_below_tol)
    self.data = data
    self.trainer_agr = trainer_agr
    self.batch_size = batch_size
    self.min_num_iter = min_num_iter
    self.max_num_iter = max_num_iter
    self.num_iter_after_best_val = num_iter_after_best_val
    self.max_num_iter_cotrain = max_num_iter_cotrain
    self.enable_summaries = enable_summaries
    self.summary_step = summary_step
    self.summary_dir = summary_dir
    self.warm_start = warm_start
    self.gradient_clip = gradient_clip
    self.logging_step = logging_step
    self.eval_step = eval_step
    self.checkpoint_path = (os.path.join(checkpoints_dir, 'classif_best.ckpt')
                            if checkpoints_dir is not None else None)
    self.weight_decay_initial = weight_decay
    self.weight_decay_schedule = weight_decay_schedule
    self.num_pairs_reg = num_pairs_reg
    self.reg_weight_ll = reg_weight_ll
    self.reg_weight_lu = reg_weight_lu
    self.reg_weight_uu = reg_weight_uu
    self.penalize_neg_agr = penalize_neg_agr
    self.use_l2_classif = use_l2_classif
    self.first_iter_original = first_iter_original
    self.iter_cotrain = iter_cotrain
    self.lr_initial = lr_initial
    self.lr_decay_steps = lr_decay_steps
    self.lr_decay_rate = lr_decay_rate
    # Build TensorFlow graph.
    logging.info('Building classification TensorFlow graph...')

    # Create placeholders.
    # First obtain the features shape from the dataset, and append a batch_size
    # dimension to it (i.e., `None` to allow for variable batch size).
    features_shape = [None] + list(data.features_shape)
    input_features = tf.placeholder(tf.float32, shape=features_shape,
                                    name='input_features')
    input_labels = tf.placeholder(tf.int64, shape=(None,), name='input_labels')
    one_hot_labels = tf.one_hot(input_labels, data.num_classes,
                                name='input_labels_one_hot')

    # Create variables and predictions.
    with tf.variable_scope('predictions'):
      encoding, variables, reg_params = self.model.get_encoding_and_params(
          inputs=input_features, is_train=is_train)
      self.variables = variables
      self.reg_params = reg_params
      predictions, variables, reg_params = (
          self.model.get_predictions_and_params(encoding=encoding,
                                                is_train=is_train))
      self.variables.update(variables)
      self.reg_params.update(reg_params)
      normalized_predictions = self.model.normalize_predictions(predictions)

    # Create a variable for weight decay that may be updated.
    weight_decay_var, weight_decay_update = self._create_weight_decay_var(
        weight_decay, weight_decay_schedule)

    # Create counter for classification iterations.
    iter_cls_total, iter_cls_total_update = self._create_counter()

    # Create loss.
    with tf.name_scope('loss'):
      if self.use_l2_classif:
        loss_supervised = tf.square(one_hot_labels - normalized_predictions)
        loss_supervised = tf.reduce_sum(loss_supervised, axis=-1)
        loss_supervised = tf.reduce_mean(loss_supervised)
      else:
        loss_supervised = self.model.get_loss(predictions=predictions,
                                              targets=one_hot_labels)

      # Agreement regularization loss.
      loss_agr = self._get_agreement_reg_loss(data, is_train, features_shape)
      # If the first co-train iteration trains the original model (for
      # comparison purposes), then we do not add an agreement loss.
      if self.first_iter_original:
        loss_agr_weight = tf.cast(tf.greater(iter_cotrain, 0), tf.float32)
        loss_agr = loss_agr * loss_agr_weight

      # Weight decay loss.
      loss_reg = 0.0
      for var in reg_params.values():
        loss_reg += weight_decay_var * tf.nn.l2_loss(var)

      # Total loss.
      loss_op = loss_supervised + loss_agr + loss_reg

    # Create accuracy.
    accuracy = tf.equal(tf.argmax(normalized_predictions, 1), input_labels)
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    # Create Tensorboard summaries.
    if self.enable_summaries:
      summaries = [
          tf.summary.scalar('loss_supervised', loss_supervised),
          tf.summary.scalar('loss_agr', loss_agr),
          tf.summary.scalar('loss_reg', loss_reg),
          tf.summary.scalar('loss_total', loss_op)
      ]
      self.summary_op = tf.summary.merge(summaries)

    # Create learning rate schedule and optimizer.
    self.global_step = tf.train.get_or_create_global_step()
    if self.lr_decay_steps is not None and self.lr_decay_rate is not None:
      self.lr = tf.train.exponential_decay(
          self.lr_initial,
          self.global_step,
          self.lr_decay_steps,
          self.lr_decay_rate,
          staircase=True)
      self.optimizer = optimizer(self.lr)
    else:
      self.optimizer = optimizer(lr_initial)

    # Get trainable variables and compute gradients.
    grads_and_vars = self.optimizer.compute_gradients(
        loss_op,
        tf.trainable_variables(scope=tf.get_default_graph().get_name_scope()))
    # Clip gradients.
    if self.gradient_clip:
      variab = [elem[1] for elem in grads_and_vars]
      gradients = [elem[0] for elem in grads_and_vars]
      gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)
      grads_and_vars = tuple(zip(gradients, variab))
    train_op = self.optimizer.apply_gradients(
        grads_and_vars, global_step=self.global_step)

    # Create a saver for model variables.
    trainable_vars = [v for _, v in grads_and_vars]

    # Put together the subset of variables to save and restore from the best
    # validation accuracy as we train the agreement model in one cotrain round.
    vars_to_save = trainable_vars
    if isinstance(weight_decay_var, tf.Variable):
      vars_to_save.append(weight_decay_var)
    saver = tf.train.Saver(vars_to_save)

    # Put together all variables that need to be saved in case the process is
    # interrupted and needs to be restarted.
    self.vars_to_save = [weight_decay_var, iter_cls_total, self.global_step]
    if self.warm_start:
      self.vars_to_save.extend([v for v in variables])

    # More variables to be initialized after the session is created.
    self.is_initialized = False

    self.rng = np.random.RandomState(seed)
    self.input_features = input_features
    self.input_labels = input_labels
    self.predictions = predictions
    self.normalized_predictions = normalized_predictions
    self.weight_decay_var = weight_decay_var
    self.weight_decay_update = weight_decay_update
    self.iter_cls_total = iter_cls_total
    self.iter_cls_total_update = iter_cls_total_update
    self.variables = variables
    self.accuracy = accuracy
    self.train_op = train_op
    self.loss_op = loss_op
    self.saver = saver
    self.batch_size_actual = tf.shape(self.predictions)[0]
    self.reset_optimizer = tf.variables_initializer(self.optimizer.variables())
    self.is_train = is_train

  def _create_weight_decay_var(self, weight_decay_initial,
                               weight_decay_schedule):
    """Creates a weight decay variable that can be updated using a schedule."""
    weight_decay_var = None
    weight_decay_update = None
    if weight_decay_schedule is None:
      weight_decay_var = tf.constant(
          weight_decay_initial, dtype=tf.float32, name='weight_decay')
    elif weight_decay_schedule == 'linear':
      weight_decay_var = tf.get_variable(
          name='weight_decay',
          initializer=tf.constant(
              weight_decay_initial, name='weight_decay_initial'),
          use_resource=True,
          trainable=False)
      update_rate = weight_decay_initial / float(self.max_num_iter_cotrain)
      weight_decay_update = weight_decay_var.assign_sub(update_rate)
    return weight_decay_var, weight_decay_update

  def _create_counter(self):
    """Creates a cummulative iteration counter for all classification steps."""
    iter_cls_total = tf.get_variable(
        name='iter_cls_total',
        initializer=tf.constant(0, name='iter_cls_total'),
        use_resource=True,
        trainable=False)
    iter_cls_total_update = iter_cls_total.assign_add(1)
    return iter_cls_total, iter_cls_total_update

  def _get_agreement_reg_loss(self, data, is_train, features_shape):
    """Computes the regularization loss coming from the agreement term.

    This is calculated using the following idea: we incur a loss for pairs of
    samples that should have the same label, but for which the predictions of
    the classification model are not equal. The loss incured by each pair is
    proportionate to the distance between the two predictions, as well as the
    confidence we have that they should agree.

    In the case of pairs where both samples are labeled (LL), the agreement
    confidence is 1.0. When at least one sample is unlabeled (LU, UU), then we
    use the agreement model to estimate this confidence.

    Note that for the pairs where a label is available, we can compute this loss
    wrt. the actual label, instead of the classifier predictions. However, when
    both samples are labeled (LL), for one of them we use the prediction and for
    the other the true label -- otherwise there are no gradients to proagate.

    Arguments:
      data: A CotrainDataset object.
      is_train: A placeholder for a boolean that specifies if this is function
        is called as part of model training or inference.
      features_shape: A tuple of integers containing the number of features in
        each dimension of the inputs, not including batch size.

    Returns:
      The computed agreement loss op.
    """
    # Select num_pairs_reg pairs of samples from each category LL, LU, UU.
    # for which to do the regularization.
    indices_lu_left = tf.placeholder(dtype=tf.int64, shape=(None,))
    indices_lu_right = tf.placeholder(dtype=tf.int64, shape=(None,))
    indices_uu_left = tf.placeholder(dtype=tf.int64, shape=(None,))
    indices_uu_right = tf.placeholder(dtype=tf.int64, shape=(None,))

    features_ll_right = tf.placeholder(dtype=tf.float32, shape=features_shape)
    features_lu_left = tf.placeholder(dtype=tf.float32, shape=features_shape)
    features_lu_right = tf.placeholder(dtype=tf.float32, shape=features_shape)
    features_uu_left = tf.placeholder(dtype=tf.float32, shape=features_shape)
    features_uu_right = tf.placeholder(dtype=tf.float32, shape=features_shape)

    labels_ll_left_idx = tf.placeholder(dtype=tf.int64, shape=(None,))
    labels_ll_right_idx = tf.placeholder(dtype=tf.int64, shape=(None,))
    labels_lu_left_idx = tf.placeholder(dtype=tf.int64, shape=(None,))

    labels_ll_left = tf.one_hot(labels_ll_left_idx, data.num_classes)
    labels_lu_left = tf.one_hot(labels_lu_left_idx, data.num_classes)

    with tf.variable_scope('predictions', reuse=True):
      encoding, _, _ = self.model.get_encoding_and_params(
          inputs=features_ll_right, is_train=is_train,
          update_batch_stats=False)
      predictions_ll_right, _, _ = self.model.get_predictions_and_params(
          encoding=encoding, is_train=is_train)
      predictions_ll_right = self.model.normalize_predictions(
          predictions_ll_right)

      encoding, _, _ = self.model.get_encoding_and_params(
          inputs=features_lu_right, is_train=is_train,
          update_batch_stats=False)
      predictions_lu_right, _, _ = self.model.get_predictions_and_params(
          encoding=encoding, is_train=is_train)
      predictions_lu_right = self.model.normalize_predictions(
          predictions_lu_right)

      encoding, _, _ = self.model.get_encoding_and_params(
          inputs=features_uu_left, is_train=is_train,
          update_batch_stats=False)
      predictions_uu_left, _, _ = self.model.get_predictions_and_params(
          encoding=encoding, is_train=is_train)
      predictions_uu_left = self.model.normalize_predictions(
          predictions_uu_left)

      encoding, _, _ = self.model.get_encoding_and_params(
          inputs=features_uu_right, is_train=is_train,
          update_batch_stats=False)
      predictions_uu_right, _, _ = self.model.get_predictions_and_params(
          encoding=encoding, is_train=is_train)
      predictions_uu_right = self.model.normalize_predictions(
          predictions_uu_right)

    # Compute Euclidean distance between the label distributions that the
    # classification model predicts for the src and tgt of each pair.
    # Stop gradients need to be added
    # The case where there are no more uu or lu
    # edges at the end of training, so the shapes don't match needs fixing.
    left = tf.concat(
        (labels_ll_left, labels_lu_left, predictions_uu_left), axis=0)
    right = tf.concat(
        (predictions_ll_right, predictions_lu_right, predictions_uu_right),
        axis=0)
    dists = tf.reduce_sum(tf.square(left - right), axis=-1)

    # Estimate a weight for each distance, depending on the predictions
    # of the agreement model. For the labeled samples, we can use the actual
    # agreement between the labels, no need to estimate.
    agreement_ll = tf.cast(
        tf.equal(labels_ll_left_idx, labels_ll_right_idx), dtype=tf.float32)
    _, agreement_lu, _, _ = self.trainer_agr.create_agreement_prediction(
        src_features=features_lu_left, tgt_features=features_lu_right,
        is_train=is_train, src_indices=indices_lu_left,
        tgt_indices=indices_lu_right)
    _, agreement_uu, _, _ = self.trainer_agr.create_agreement_prediction(
        src_features=features_uu_left, tgt_features=features_uu_right,
        is_train=is_train, src_indices=indices_uu_left,
        tgt_indices=indices_uu_right)
    agreement = tf.concat((agreement_ll, agreement_lu, agreement_uu), axis=0)
    if self.penalize_neg_agr:
      # Since the agreement is predicting scores between [0, 1], anything
      # under 0.5 should represent disagreement. Therefore, we want to encourage
      # agreement whenever the score is > 0.5, otherwise don't incurr any loss.
      agreement = tf.nn.relu(agreement - 0.5)

      # Create a Tensor containing the weights assigned to each pair in the
    # agreement regularization loss, depending on how many samples in the pair
    # were labeled. This weight can be either reg_weight_ll, reg_weight_lu,
    # or reg_weight_uu.
    num_ll = tf.shape(predictions_ll_right)[0]
    num_lu = tf.shape(predictions_lu_right)[0]
    num_uu = tf.shape(predictions_uu_left)[0]
    weights = tf.concat((self.reg_weight_ll * tf.ones(num_ll,),
                         self.reg_weight_lu * tf.ones(num_lu,),
                         self.reg_weight_uu * tf.ones(num_uu,)),
                        axis=0)

    # Scale each distance by its agreement weight and regularzation weight.
    loss = tf.reduce_mean(dists * weights * agreement)

    self.indices_lu_left = indices_lu_left
    self.indices_lu_right = indices_lu_right
    self.indices_uu_left = indices_uu_left
    self.indices_uu_right = indices_uu_right
    self.features_ll_right = features_ll_right
    self.features_lu_left = features_lu_left
    self.features_lu_right = features_lu_right
    self.features_uu_left = features_uu_left
    self.features_uu_right = features_uu_right
    self.labels_ll_left = labels_ll_left_idx
    self.labels_ll_right = labels_ll_right_idx
    self.labels_lu_left = labels_lu_left_idx
    self.agreement_lu = agreement_lu

    return loss

  def _construct_feed_dict(self,
                           data_iterator,
                           is_train,
                           pair_ll_iterator=None,
                           pair_lu_iterator=None,
                           pair_uu_iterator=None):
    """Construct feed dictionary."""
    try:
      input_indices = next(data_iterator)
      # Select the labels. Use the true, correct labels, at test time, and the
      # self-labeled ones at train time.
      labels = (self.data.get_labels(input_indices) if is_train else
                self.data.get_original_labels(input_indices))
      feed_dict = {
          self.input_features: self.data.get_features(input_indices),
          self.input_labels: labels,
          self.is_train: is_train
      }
      if pair_ll_iterator is not None:
        _, _, _, features_tgt, labels_src, labels_tgt = next(pair_ll_iterator)
        feed_dict.update({
            self.features_ll_right: features_tgt,
            self.labels_ll_left: labels_src,
            self.labels_ll_right: labels_tgt
        })
      if pair_lu_iterator is not None:
        indices_src, indices_tgt, features_src, features_tgt, labels_src, _ = (
            next(pair_lu_iterator))
        feed_dict.update({
            self.indices_lu_left: indices_src,
            self.indices_lu_right: indices_tgt,
            self.features_lu_left: features_src,
            self.features_lu_right: features_tgt,
            self.labels_lu_left: labels_src
        })
      if pair_uu_iterator is not None:
        indices_src, indices_tgt, features_src, features_tgt, _, _ = next(
            pair_uu_iterator)
        feed_dict.update({
            self.indices_uu_left: indices_src,
            self.indices_uu_right: indices_tgt,
            self.features_uu_left: features_src,
            self.features_uu_right: features_tgt
        })
      return feed_dict
    except StopIteration:
      # If the iterator has finished, return None.
      return None

  def pair_iterator(self, src_indices, tgt_indices, batch_size, data):
    """Iterator over pairs of samples.

    The first element of the pair is selected from the src_indices, and the
    second element is selected from tgt_indices.

    Arguments:
      src_indices: Numpy array containing the indices available for the source
        node.
      tgt_indices: Numpy array containing the indices available for the tgt
        node.
      batch_size: An integer representing the desired batch size.
      data: A CotrainDataset object used to extract the features and labels.

    Yields:
      indices_src, indices_tgt, features_src, features_tgt, labels_src,
      labels_tgt
    """

    def _select_from_pool(indices):
      """Selects batch_size indices from the provided list."""
      num_indices = len(indices)
      if num_indices > 0:
        idxs = self.rng.randint(0, high=num_indices, size=(batch_size,))
        indices_batch = indices[idxs]
        features_batch = data.get_features(indices_batch)
        labels_batch = data.get_labels(indices_batch)
      else:
        features_shape = [0] + list(data.features_shape)
        indices_batch = np.zeros(shape=(0,), dtype=np.int64)
        features_batch = np.zeros(shape=features_shape, dtype=np.float32)
        labels_batch = np.zeros(shape=(0,), dtype=np.int64)
      return indices_batch, features_batch, labels_batch

    while True:
      indices_src, features_src, labels_src = _select_from_pool(src_indices)
      indices_tgt, features_tgt, labels_tgt = _select_from_pool(tgt_indices)
      yield (indices_src, indices_tgt, features_src, features_tgt,
             labels_src, labels_tgt)

  def train(self, data, session=None, **kwargs):
    """Train the classification model on the provided dataset.

    Arguments:
      data: A CotrainDataset object.
      session: A TensorFlow session or None.
      **kwargs: Other keyword arguments.
    Returns:
      best_test_acc: A float representing the test accuracy at the iteration
        where the validation accuracy is maximum.
      best_val_acc: A float representing the best validation accuracy.
    """
    summary_writer = kwargs['summary_writer']
    logging.info('Training classifier...')

    if not self.is_initialized:
      self.is_initialized = True
      logging.info('Weight decay value: %f', session.run(self.weight_decay_var))
    else:
      if self.weight_decay_update is not None:
        session.run(self.weight_decay_update)
        logging.info('New weight decay value:  %f',
                     session.run(self.weight_decay_var))
      # Reset the optimizer state (e.g., momentum).
      session.run(self.reset_optimizer)

    if not self.warm_start:
      # Re-initialize variables.
      initializers = [v.initializer for v in self.variables.values()]
      initializers.append(self.global_step.initializer)
      session.run(initializers)

    # Construct data iterator.
    logging.info('Training classifier with %d samples...', data.num_train())
    train_indices = data.get_indices_train()
    unlabeled_indices = data.get_indices_unlabeled()
    val_indices = data.get_indices_val()
    test_indices = data.get_indices_test()
    # Create an iterator for labeled samples for the supervised term.
    data_iterator_train = batch_iterator(
        train_indices,
        batch_size=self.batch_size,
        shuffle=True,
        allow_smaller_batch=False,
        repeat=True)
    # Create iterators for ll, lu, uu pairs of samples for the agreement term.
    pair_ll_iterator = self.pair_iterator(
        train_indices, train_indices, self.num_pairs_reg, data)
    pair_lu_iterator = self.pair_iterator(
        train_indices, unlabeled_indices, self.num_pairs_reg, data)
    pair_uu_iterator = self.pair_iterator(
        unlabeled_indices, unlabeled_indices, self.num_pairs_reg, data)

    step = 0
    iter_below_tol = 0
    min_num_iter = self.min_num_iter
    has_converged = step >= self.max_num_iter
    prev_loss_val = np.inf
    best_test_acc = -1
    best_val_acc = -1
    checkpoint_saved = False
    while not has_converged:
      feed_dict = self._construct_feed_dict(data_iterator_train, True,
                                            pair_ll_iterator, pair_lu_iterator,
                                            pair_uu_iterator)
      if self.enable_summaries and step % self.summary_step == 0:
        loss_val, summary, _ = session.run(
            [self.loss_op, self.summary_op, self.train_op],
            feed_dict=feed_dict)
        iter_cls_total = session.run(self.iter_cls_total)
        summary_writer.add_summary(summary, iter_cls_total)
        summary_writer.flush()
      else:
        loss_val, _ = session.run((self.loss_op, self.train_op),
                                  feed_dict=feed_dict)

      # Log the loss, if necessary.
      if step % self.logging_step == 0:
        logging.info('Classification step %6d | Loss: %10.4f', step, loss_val)

      # Evaluate, if necessary.
      def _evaluate(indices, name):
        """Evaluates the samples with the provided indices."""
        data_iterator_val = batch_iterator(
            indices,
            batch_size=self.batch_size,
            shuffle=False,
            allow_smaller_batch=True,
            repeat=False)
        feed_dict_val = self._construct_feed_dict(data_iterator_val, False)
        cummulative_acc = 0.0
        num_samples = 0
        while feed_dict_val is not None:
          val_acc, batch_size_actual = session.run(
              (self.accuracy, self.batch_size_actual), feed_dict=feed_dict_val)
          cummulative_acc += val_acc * batch_size_actual
          num_samples += batch_size_actual
          feed_dict_val = self._construct_feed_dict(data_iterator_val, False)
        if num_samples > 0:
          cummulative_acc /= num_samples

        if self.enable_summaries:
          summary = tf.Summary()
          summary.value.add(
              tag='ClassificationModel/' + name + '_acc',
              simple_value=cummulative_acc)
          iter_cls_total = session.run(self.iter_cls_total)
          summary_writer.add_summary(summary, iter_cls_total)
          summary_writer.flush()

        return cummulative_acc

      # Run validation, if necessary.
      if step % self.eval_step == 0:
        logging.info('Evaluating on %d validation samples...', len(val_indices))
        val_acc = _evaluate(val_indices, 'val_acc')
        logging.info('Evaluating on %d test samples...', len(test_indices))
        test_acc = _evaluate(test_indices, 'test_acc')

        if step % self.logging_step == 0 or val_acc > best_val_acc:
          logging.info(
              'Classification step %6d | Loss: %10.4f | '
              'val_acc: %10.4f | test_acc: %10.4f', step, loss_val, val_acc,
              test_acc)
        if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_test_acc = test_acc
          if self.checkpoint_path:
            self.saver.save(
                session, self.checkpoint_path, write_meta_graph=False)
            checkpoint_saved = True
          # Go for at least num_iter_after_best_val more iterations.
          min_num_iter = max(self.min_num_iter,
                             step + self.num_iter_after_best_val)
          logging.info(
              'Achieved best validation. '
              'Extending to at least %d iterations...', min_num_iter)

      step += 1
      has_converged, iter_below_tol = self.check_convergence(
          prev_loss_val,
          loss_val,
          step,
          self.max_num_iter,
          iter_below_tol,
          min_num_iter=min_num_iter)
      session.run(self.iter_cls_total_update)
      prev_loss_val = loss_val

    # Return to the best model.
    if checkpoint_saved:
      logging.info('Restoring best model...')
      self.saver.restore(session, self.checkpoint_path)

    return best_test_acc, best_val_acc

  def predict(self, session, indices):
    """Make predictions for the provided sample indices."""
    num_inputs = len(indices)
    idx_start = 0
    predictions = []
    while idx_start < num_inputs:
      idx_end = min(idx_start + self.batch_size, num_inputs)
      batch_indices = indices[idx_start:idx_end]
      input_features = self.data.get_features(batch_indices)
      batch_predictions = session.run(
          self.normalized_predictions,
          feed_dict={self.input_features: input_features})
      predictions.append(batch_predictions)
      idx_start = idx_end
    if not predictions:
      return np.zeros((0, self.data.num_classes), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


class TrainerPerfectClassification(Trainer):
  """Trainer for a classifier that always predicts the correct value."""

  def __init__(self, data):
    self.data = data

  def train(self, unused_data, unused_session=None, **unused_kwargs):
    logging.info('Perfect classifier, no need to train...')
    return 1.0, 1.0

  def predict(self, unused_session, indices_unlabeled):
    return self.data.get_original_labels(indices_unlabeled)
