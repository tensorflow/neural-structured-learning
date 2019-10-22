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
"""Trains a classification model and a label agreement model using co-training.

This class makes use of a Trainer for the classification model and a trainer
for the agreement model, and alternatively trains each of them. After each
iteration some unlabeled samples are labeled using the classification model,
such that in the next iteration both models are re-trained using more labeled
data.

Throughout this file, the suffix "_cls" refers to the classification model, and
"_agr" to the agreement model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from gam.data.dataset import CotrainDataset
from gam.models.gcn import GCN
from gam.trainer.trainer_agreement import TrainerAgreement
from gam.trainer.trainer_agreement import TrainerAgreementAlwaysAgree
from gam.trainer.trainer_agreement import TrainerPerfectAgreement
from gam.trainer.trainer_base import Trainer
from gam.trainer.trainer_classification import TrainerClassification
from gam.trainer.trainer_classification import TrainerPerfectClassification
from gam.trainer.trainer_classification_gcn import TrainerClassificationGCN

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class TrainerCotraining(Trainer):
  """Trainer for a co-training model with agreement.

  Attributes:
    model_cls: An object whose type is a subclass of Model, representing the
      model for the sample classifier.
    model_agr: An object whose type is a subclass of Model, representing the
      model for the agreement model.
    max_num_iter_cotrain: An integer representing the maximum number of cotrain
      iterations to perform.
    min_num_iter_cls: An integer representing the minimum number of iterations
      to train the classification model for.
    max_num_iter_cls: An integer representing the maximum number of iterations
      to train the classification model for.
    num_iter_after_best_val_cls: An integer representing the number of extra
      iterations to perform after improving the validation accuracy of the
      classification model.
    min_num_iter_agr: An integer representing the minimum number of iterations
      to train the agreement model for.
    max_num_iter_agr: An integer representing the maximum number of iterations
      to train the agreement model.
    num_iter_after_best_val_agr: An integer representing the number of extra
      iterations to perform after improving the agreement validation accuracy.
    num_samples_to_label: Maximum number of samples to self-label after each
      cotrain iteration, provided that they have confidence higher than the
      min_confidence_new_label threshold.
    min_confidence_new_label: A float number between [0, 1] representing the
      minimum confidence the prediction for an unlabeled sample needs to have in
      order to allow it to be self-labeled. The confidence is the maximum
      probability the classification model assigns to any of the classes.
    keep_label_proportions: A boolean specifying whether to choose samples for
      self-labeling such that we maintain the original label proportions.
    num_warm_up_iter_agr: An integer representing the number of times we need to
      train the agreement model (i.e. number of cotrain iterations that train
      the agreement) before we start using it in the classification model's
      loss. While the agreement is not warmed up, the agreement model will
      always predict either disagreement, or agreement, by default, depending on
      the argument `agree_by_default`.
    optimizer: An optimizer.
    gradient_clip: A float number representing the maximum gradient norm allowed
      if we do gradient clipping. If None, no gradient clipping is performed.
    batch_size_agr: An integer representing the batch size of the agreement
      model.
    batch_size_cls: An integer representing the batch size of the classification
      model. This is used for the supervised component of the loss and for
      evaluation.
    learning_rate_cls: A float representing the learning rate used when training
      the classification model.
    learning_rate_agr: A float representing the learning rate used when training
      the agreement model.
    warm_start_cls: Boolean specifying if the classification model is trained
      from scratch in every cotrain itertion (if False), or if it continues from
      the parameter values in the previous cotrain iteration (if True).
    warm_start_agr: Boolean specifying if the agreement model is trained from
      scratch in every cotrain itertion (if False), or if it continues from the
      parameter values in the previous cotrain iteration (if True).
    enable_summaries: Boolean specifying whether to write TensorBoard summaries
      for the cotrain progress.
    enable_summaries_per_model: Boolean specifying whether to write TensorBoard
      summaries for the classification and agreement model progress.
    summary_dir: Directory path where to save the Tensorflow summaries.
    summary_step_cls: Integer representing the number of iterations after which
      to write TensorFlow summaries for the classification model.
    summary_step_agr: Integer representing the number of iterations after which
      to write TensorFlow summaries for the agreement model.
    logging_step_cls: Integer representing the number of iterations after which
      to log the loss and other training metrics for the classification model.
    logging_step_agr: Integer representing the number of iterations after which
      to log the loss and other training metrics for the agreement model.
    eval_step_cls: Integer representing the number of iterations after which to
      evaluate the classification model.
    eval_step_agr: Integer representing the number of iterations after which to
      evaluate the agreement model.
    checkpoints_step: Integer representing the number of iterations after which
      to save checkpoints.
    checkpoints_dir: Directory where to save checkpoints.
    data_dir: Directory where to write some files that contain self-labeled data
      backup.
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
    use_perfect_agr: Boolean specifying whether to use a perfect agreement model
      that peeks at the correct test labels (for debugging only).
    use_perfect_cls: Boolean specifying whether to use a perfect classification
      model that peeks at the correct test labels (for debugging only).
    ratio_valid_agr: Ratio of the labeled sample pairs to use for validation
      whent training the agreement model.
    max_samples_valid_agr: Maximum number of sample pairs to use for validation
      whent training the agreement model.
    weight_decay_cls: Weight for the weight decay term in the classification
      model loss.
    weight_decay_schedule_cls: Schedule how to adjust the classification weight
      decay weight after every cotrain iteration.
    weight_decay_agr: Weight for the weight decay term in the agreement model
      loss.
    weight_decay_schedule_agr: Schedule how to adjust the agreement weight decay
      weight after every cotrain iteration.
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
    reg_weight_vat: A float representing the weight of the virtual adversarial
      training (VAT) regularization loss in the classification model loss
      function.
    use_ent_min: A boolean specifying whether to use entropy regularization with
      VAT.
    penalize_neg_agr: Whether to not only encourage agreement between samples
      that the agreement model believes should have the same label, but also
      penalize agreement when two samples agree when the agreement model
      predicts they should disagree.
    use_l2_cls: Whether to use L2 loss for classification, as opposed to the
      whichever loss is specified in the provided model_cls.
    first_iter_original: A boolean specifying whether the first cotrain
      iteration trains the original classification model (with no agreement
      term). We do this to evaluate how well a baseline model would do without
      the agreement. If true, there is no self-labeling after the first
      iteration, which trains original model. Self-labeling will be used only in
      the iterations that do include the agreement term.
    inductive: Boolean specifying whether this is an inductive or transductive
      setting. If inductive, then the validation and test labels are never seen
      when training the classification model. If transductive, the inputs of the
      test and validation samples are available at training time and can be used
      in the agreement loss term of the classification model as unsupervised
      regularization, and can also be labeled via self-labeling.
    seed: An integer representing the seed for the random number generator used
      when selecting batches of samples.
    eval_acc_pred_by_agr: Boolean specifying whether to evaluate the accuracy of
      a model that uses our trained agreement model to make predictions for the
      test samples, in a way similar to k-nearest neighbors, where the distance
      is given by the agreement model predictions.
    num_neighbors_pred_by_agr: An integer representing the number of neighbors
      to use when predicting by agreement. Note that this needs to be at least
      as much as the number of classes.
    load_from_checkpoint: A boolean specifying whethe the trained models are
      loaded from checkpoint, if one is available. If False, the models are
      always trained from scratch.
    use_graph: Boolean specifying whether to use to apply the agreement model on
      the graph edges, or otherwise use random pairs of samples.
    always_agree: Whether the agreement model should return 1.0 always (i.e. the
      samples always agree), to simulate the Neural Graph Machines model.
    add_negative_edges_agr:
  """

  def __init__(self,
               model_cls,
               model_agr,
               max_num_iter_cotrain,
               min_num_iter_cls,
               max_num_iter_cls,
               num_iter_after_best_val_cls,
               min_num_iter_agr,
               max_num_iter_agr,
               num_iter_after_best_val_agr,
               num_samples_to_label,
               min_confidence_new_label=0.0,
               keep_label_proportions=False,
               num_warm_up_iter_agr=1,
               optimizer=tf.train.AdamOptimizer,
               gradient_clip=None,
               batch_size_agr=128,
               batch_size_cls=128,
               learning_rate_cls=1e-3,
               learning_rate_agr=1e-3,
               warm_start_cls=False,
               warm_start_agr=False,
               enable_summaries=True,
               enable_summaries_per_model=False,
               summary_dir=None,
               summary_step_cls=1000,
               summary_step_agr=1000,
               logging_step_cls=1,
               logging_step_agr=1,
               eval_step_cls=1,
               eval_step_agr=1,
               checkpoints_step=None,
               checkpoints_dir=None,
               data_dir=None,
               abs_loss_chg_tol=1e-10,
               rel_loss_chg_tol=1e-7,
               loss_chg_iter_below_tol=30,
               use_perfect_agr=False,
               use_perfect_cls=False,
               ratio_valid_agr=0,
               max_samples_valid_agr=None,
               weight_decay_cls=None,
               weight_decay_schedule_cls=None,
               weight_decay_agr=None,
               weight_decay_schedule_agr=None,
               reg_weight_ll=0,
               reg_weight_lu=0,
               reg_weight_uu=0,
               num_pairs_reg=100,
               reg_weight_vat=0,
               use_ent_min=False,
               penalize_neg_agr=False,
               use_l2_cls=True,
               first_iter_original=True,
               inductive=False,
               seed=None,
               eval_acc_pred_by_agr=False,
               num_neighbors_pred_by_agr=20,
               lr_decay_rate_cls=None,
               lr_decay_steps_cls=None,
               lr_decay_rate_agr=None,
               lr_decay_steps_agr=None,
               load_from_checkpoint=False,
               use_graph=False,
               always_agree=False,
               add_negative_edges_agr=False):
    assert not enable_summaries or (enable_summaries and
                                    summary_dir is not None)
    assert checkpoints_step is None or (checkpoints_step is not None and
                                        checkpoints_dir is not None)
    super(TrainerCotraining, self).__init__(
        model=None,
        abs_loss_chg_tol=abs_loss_chg_tol,
        rel_loss_chg_tol=rel_loss_chg_tol,
        loss_chg_iter_below_tol=loss_chg_iter_below_tol)
    self.model_cls = model_cls
    self.model_agr = model_agr
    self.max_num_iter_cotrain = max_num_iter_cotrain
    self.min_num_iter_cls = min_num_iter_cls
    self.max_num_iter_cls = max_num_iter_cls
    self.num_iter_after_best_val_cls = num_iter_after_best_val_cls
    self.min_num_iter_agr = min_num_iter_agr
    self.max_num_iter_agr = max_num_iter_agr
    self.num_iter_after_best_val_agr = num_iter_after_best_val_agr
    self.num_samples_to_label = num_samples_to_label
    self.min_confidence_new_label = min_confidence_new_label
    self.keep_label_proportions = keep_label_proportions
    self.num_warm_up_iter_agr = num_warm_up_iter_agr
    self.optimizer = optimizer
    self.gradient_clip = gradient_clip
    self.batch_size_agr = batch_size_agr
    self.batch_size_cls = batch_size_cls
    self.learning_rate_cls = learning_rate_cls
    self.learning_rate_agr = learning_rate_agr
    self.warm_start_cls = warm_start_cls
    self.warm_start_agr = warm_start_agr
    self.enable_summaries = enable_summaries
    self.enable_summaries_per_model = enable_summaries_per_model
    self.summary_step_cls = summary_step_cls
    self.summary_step_agr = summary_step_agr
    self.summary_dir = summary_dir
    self.logging_step_cls = logging_step_cls
    self.logging_step_agr = logging_step_agr
    self.eval_step_cls = eval_step_cls
    self.eval_step_agr = eval_step_agr
    self.checkpoints_step = checkpoints_step
    self.checkpoints_dir = checkpoints_dir
    self.data_dir = data_dir
    self.use_perfect_agr = use_perfect_agr
    self.use_perfect_cls = use_perfect_cls
    self.ratio_valid_agr = ratio_valid_agr
    self.max_samples_valid_agr = max_samples_valid_agr
    self.weight_decay_cls = weight_decay_cls
    self.weight_decay_schedule_cls = weight_decay_schedule_cls
    self.weight_decay_agr = weight_decay_agr
    self.weight_decay_schedule_agr = weight_decay_schedule_agr
    self.reg_weight_ll = reg_weight_ll
    self.reg_weight_lu = reg_weight_lu
    self.reg_weight_uu = reg_weight_uu
    self.num_pairs_reg = num_pairs_reg
    self.reg_weight_vat = reg_weight_vat
    self.use_ent_min = use_ent_min
    self.penalize_neg_agr = penalize_neg_agr
    self.use_l2_classif = use_l2_cls
    self.first_iter_original = first_iter_original
    self.inductive = inductive
    self.seed = seed
    self.eval_acc_pred_by_agr = eval_acc_pred_by_agr
    self.num_neighbors_pred_by_agr = num_neighbors_pred_by_agr
    self.lr_decay_rate_cls = lr_decay_rate_cls
    self.lr_decay_steps_cls = lr_decay_steps_cls
    self.lr_decay_rate_agr = lr_decay_rate_agr
    self.lr_decay_steps_agr = lr_decay_steps_agr
    self.load_from_checkpoint = load_from_checkpoint
    self.use_graph = use_graph
    self.always_agree = always_agree
    self.add_negative_edges_agr = add_negative_edges_agr

  def _select_samples_to_label(self, data, trainer_cls, session):
    """Selects which samples to label next.

    Arguments:
      data: A CotrainData object.
      trainer_cls: A TrainerClassification object.
      session: A TensorFlow Session.

    Returns:
      selected_samples: numpy array containing the indices of the samples to be
        labeled.
      selected_labels: numpy array containing the indices of the labels to
        assign to each of the selected nodes.
    """
    # Select the candidate samples for self-labeling, and make predictions.
    # Remove the validation samples from the unlabeled data, if there, to avoid
    # self-labeling them.
    indices_unlabeled = data.get_indices_unlabeled()
    val_ind = set(data.get_indices_val())
    indices_unlabeled = np.asarray(
        [ind for ind in indices_unlabeled if ind not in val_ind])
    predictions = trainer_cls.predict(
        session, indices_unlabeled, is_train=False)

    # Select most confident nodes. Compute confidence and most confident label,
    # which will be used as the new label.
    predicted_label = np.argmax(predictions, axis=-1)
    confidence = predictions[np.arange(predicted_label.shape[0]),
                             predicted_label]
    # Sort from most confident to least confident.
    indices_sorted = np.argsort(confidence)[::-1]
    indices_unlabeled = indices_unlabeled[indices_sorted]
    confidence = confidence[indices_sorted]
    predicted_label = predicted_label[indices_sorted]

    # Keep only samples that have at least min_confidence_new_label confidence.
    confident_indices = np.argwhere(
        confidence > self.min_confidence_new_label)[:, 0]
    if confident_indices.shape[0] == 0:
      logging.info(
          'No unlabeled nodes with confidence > %.2f. '
          'Skipping self-labeling...', self.min_confidence_new_label)
      selected_samples = np.zeros((0,), dtype=np.int64)
      selected_labels = np.zeros((0,), dtype=np.int64)
      return selected_samples, selected_labels

    if data.keep_label_proportions:
      # Pick the top num_samples_to_label most confident nodes, while making
      # sure the ratio of the labels are kept.
      # First keep only nodes which achieve the min required confidence.
      num_confident = len(confident_indices)
      nodes_with_min_conf = indices_unlabeled[:num_confident]
      labels_with_min_conf = predicted_label[:num_confident]
      # Out of these, select the desired number of samples per class,
      # according to class proportions.
      selected_samples = []
      selected_labels = []
      for label, prop in data.label_prop.items():
        num_samples_to_select = int(prop * self.num_samples_to_label)
        label_idxs = np.where(labels_with_min_conf == label)[0]
        if len(label_idxs) <= num_samples_to_select:
          # Select all available samples labeled with this label.
          selected_samples.append(nodes_with_min_conf[label_idxs])
          selected_labels.append(labels_with_min_conf[label_idxs])
        elif num_samples_to_select > 0:
          # Select the first ones, since they are sorted by confidence.
          selected_samples.append(
              nodes_with_min_conf[label_idxs][:num_samples_to_select])
          selected_labels.append(
              labels_with_min_conf[label_idxs][:num_samples_to_select])
      selected_samples = np.concatenate(selected_samples)
      selected_labels = np.concatenate(selected_labels)
    else:
      # Pick the top num_samples_to_label most confident nodes,
      # irrespective of their labels.
      idx = np.amax(confident_indices)
      max_idx = min(self.num_samples_to_label - 1, idx)
      selected_samples = indices_unlabeled[:max_idx + 1]
      selected_labels = predicted_label[:max_idx + 1]

    return selected_samples, selected_labels

  def _extend_label_set(self, data, trainer_cls, session):
    """Extend labeled set by self-labeling with most confident predictions."""
    # Select which nodes to label next, and predict their labels.
    selected_samples, selected_labels = self._select_samples_to_label(
        data, trainer_cls, session)
    # Replace the labels of the new nodes with the predicted labels.
    if selected_samples.shape[0] > 0:
      data.label_samples(selected_samples, selected_labels)
    return selected_samples

  def train(self, data, **kwargs):
    # Create a wrapper around the dataset, that also accounts for some
    # cotrain specific attributes and functions.
    data = CotrainDataset(
        data,
        keep_label_proportions=self.keep_label_proportions,
        inductive=self.inductive)

    if os.path.exists(self.data_dir) and self.load_from_checkpoint:
      # If this session is restored from a previous run, then we load the
      # self-labeled data from the last checkpoint.
      logging.info('Number of labeled samples before restoring: %d',
                   data.num_train())
      logging.info('Restoring self-labeled data from %s...', self.data_dir)
      data.restore_state_from_file(self.data_dir)
      logging.info('Number of labeled samples after restoring: %d',
                   data.num_train())

    # Build graph.
    logging.info('Building graph...')

    # Create a iteration counter.
    iter_cotrain, iter_cotrain_update = self._create_counter()

    if self.use_perfect_agr:
      # A perfect agreement model used for model.
      trainer_agr = TrainerPerfectAgreement(data=data)
    else:
      with tf.variable_scope('AgreementModel'):
        if self.always_agree:
          trainer_agr = TrainerAgreementAlwaysAgree(data=data)
        else:
          trainer_agr = TrainerAgreement(
              model=self.model_agr,
              data=data,
              optimizer=self.optimizer,
              gradient_clip=self.gradient_clip,
              min_num_iter=self.min_num_iter_agr,
              max_num_iter=self.max_num_iter_agr,
              num_iter_after_best_val=self.num_iter_after_best_val_agr,
              max_num_iter_cotrain=self.max_num_iter_cotrain,
              num_warm_up_iter=self.num_warm_up_iter_agr,
              warm_start=self.warm_start_agr,
              batch_size=self.batch_size_agr,
              enable_summaries=self.enable_summaries_per_model,
              summary_step=self.summary_step_agr,
              summary_dir=self.summary_dir,
              logging_step=self.logging_step_agr,
              eval_step=self.eval_step_agr,
              abs_loss_chg_tol=self.abs_loss_chg_tol,
              rel_loss_chg_tol=self.rel_loss_chg_tol,
              loss_chg_iter_below_tol=self.loss_chg_iter_below_tol,
              checkpoints_dir=self.checkpoints_dir,
              weight_decay=self.weight_decay_agr,
              weight_decay_schedule=self.weight_decay_schedule_agr,
              agree_by_default=False,
              percent_val=self.ratio_valid_agr,
              max_num_samples_val=self.max_samples_valid_agr,
              seed=self.seed,
              lr_decay_rate=self.lr_decay_rate_agr,
              lr_decay_steps=self.lr_decay_steps_agr,
              lr_initial=self.learning_rate_agr,
              use_graph=self.use_graph,
              add_negative_edges=self.add_negative_edges_agr)

    if self.use_perfect_cls:
      # A perfect classification model used for debugging purposes.
      trainer_cls = TrainerPerfectClassification(data=data)
    else:
      with tf.variable_scope('ClassificationModel'):
        trainer_cls_class = (
          TrainerClassificationGCN if isinstance(self.model_cls, GCN) else
          TrainerClassification)
        trainer_cls = trainer_cls_class(
            model=self.model_cls,
            data=data,
            trainer_agr=trainer_agr,
            optimizer=self.optimizer,
            gradient_clip=self.gradient_clip,
            batch_size=self.batch_size_cls,
            min_num_iter=self.min_num_iter_cls,
            max_num_iter=self.max_num_iter_cls,
            num_iter_after_best_val=self.num_iter_after_best_val_cls,
            max_num_iter_cotrain=self.max_num_iter_cotrain,
            reg_weight_ll=self.reg_weight_ll,
            reg_weight_lu=self.reg_weight_lu,
            reg_weight_uu=self.reg_weight_uu,
            num_pairs_reg=self.num_pairs_reg,
            reg_weight_vat=self.reg_weight_vat,
            use_ent_min=self.use_ent_min,
            enable_summaries=self.enable_summaries_per_model,
            summary_step=self.summary_step_cls,
            summary_dir=self.summary_dir,
            logging_step=self.logging_step_cls,
            eval_step=self.eval_step_cls,
            abs_loss_chg_tol=self.abs_loss_chg_tol,
            rel_loss_chg_tol=self.rel_loss_chg_tol,
            loss_chg_iter_below_tol=self.loss_chg_iter_below_tol,
            warm_start=self.warm_start_cls,
            checkpoints_dir=self.checkpoints_dir,
            weight_decay=self.weight_decay_cls,
            weight_decay_schedule=self.weight_decay_schedule_cls,
            penalize_neg_agr=self.penalize_neg_agr,
            use_l2_classif=self.use_l2_classif,
            first_iter_original=self.first_iter_original,
            seed=self.seed,
            iter_cotrain=iter_cotrain,
            lr_decay_rate=self.lr_decay_rate_cls,
            lr_decay_steps=self.lr_decay_steps_cls,
            lr_initial=self.learning_rate_cls,
            use_graph=self.use_graph)

    # Create a saver which saves only the variables that we would need to
    # restore in case the training process is restarted.
    vars_to_save = [iter_cotrain] + trainer_agr.vars_to_save + \
                   trainer_cls.vars_to_save
    saver = tf.train.Saver(vars_to_save)

    # Create a TensorFlow session. We allow soft placement in order to place
    # any supported ops on GPU. The allow_growth option lets our process
    # progressively use more gpu memory, per need basis, as opposed to
    # allocating it all from the beginning.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Create a Tensorflow summary writer, shared by all models.
    summary_writer = tf.summary.FileWriter(self.summary_dir, session.graph)

    # Initialize the values of all variables and the train dataset iterator.
    session.run(tf.global_variables_initializer())

    # If a checkpoint with the variables already exists, we restore them.
    if self.checkpoints_dir:
      checkpts_path_cotrain = os.path.join(self.checkpoints_dir, 'cotrain.ckpt')
      if os.path.exists(checkpts_path_cotrain):
        if self.load_from_checkpoint:
          saver.restore(session, checkpts_path_cotrain)
      else:
        os.makedirs(checkpts_path_cotrain)
    else:
      checkpts_path_cotrain = None

    # Create a progress bar showing how many samples are labeled.
    pbar = tqdm(
        total=data.num_samples - data.num_train(), desc='self-labeled nodes')

    logging.info('Starting co-training...')
    step = session.run(iter_cotrain)
    stop = step >= self.max_num_iter_cotrain
    best_val_acc = -1
    test_acc_at_best = -1
    iter_at_best = -1
    while not stop:
      logging.info('----------------- Cotrain step %6d -----------------', step)
      # Train the agreement model.
      if self.first_iter_original and step == 0:
        logging.info('First iteration trains the original classifier.'
                     'No need to train the agreement model.')
        val_acc_agree = None
        acc_pred_by_agr = None
      else:
        val_acc_agree = trainer_agr.train(
            data, session=session, summary_writer=summary_writer)

        if self.eval_acc_pred_by_agr:
          # Evaluate the prediction accuracy by a majority vote model using the
          # agreement model.
          logging.info('Computing agreement majority vote predictions on '
                       'test data...')
          acc_pred_by_agr = trainer_agr.predict_label_by_agreement(
              session, data.get_indices_test(), self.num_neighbors_pred_by_agr)
        else:
          acc_pred_by_agr = None

      # Train classification model.
      test_acc, val_acc = trainer_cls.train(
          data, session=session, summary_writer=summary_writer)

      if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc_at_best = test_acc
        iter_at_best = step

      if self.enable_summaries:
        summary = tf.Summary()
        summary.value.add(tag='cotrain/test_acc', simple_value=test_acc)
        summary.value.add(tag='cotrain/val_acc', simple_value=val_acc)
        if val_acc_agree is not None:
          summary.value.add(
              tag='cotrain/val_acc_agree', simple_value=val_acc_agree)
        if acc_pred_by_agr is not None:
          summary.value.add(
              tag='cotrain/acc_predict_by_agreement',
              simple_value=acc_pred_by_agr)
        summary_writer.add_summary(summary, step)
        summary_writer.flush()

      logging.info(
          '--------- Cotrain step %6d | Accuracy val: %10.4f | '
          'Accuracy test: %10.4f ---------', step, val_acc, test_acc)

      if self.first_iter_original and step == 0:
        logging.info('No self-labeling because the first iteration trains the '
                     'original classifier for evaluation purposes.')
        step += 1
      else:
        # Extend labeled set by self-labeling.
        logging.info('Self-labeling...')
        selected_samples = self._extend_label_set(data, trainer_cls, session)

        # If no new data points are added to the training set, stop.
        num_new_labels = len(selected_samples)
        pbar.update(num_new_labels)
        if num_new_labels > 0:
          data.compute_dataset_statistics(selected_samples, summary_writer,
                                          step)
        else:
          logging.info('No new samples labeled. Stopping...')
          stop = True

        step += 1
        stop |= step >= self.max_num_iter_cotrain

        # Save model and dataset state in case of process preemption.
        if self.checkpoints_step and step % self.checkpoints_step == 0:
          self._save_state(saver, session, data, checkpts_path_cotrain)

      session.run(iter_cotrain_update)
      logging.info('________________________________________________________')

    logging.info(
        'Best validation acc: %.4f, corresponding test acc: %.4f at '
        'iteration %d', best_val_acc, test_acc_at_best, iter_at_best)
    pbar.close()

  def _create_counter(self):
    """Creates a cotrain iteration counter."""
    iter_cotrain = tf.get_variable(
        name='iter_cotrain',
        initializer=tf.constant(0, name='iter_cotrain'),
        use_resource=True,
        trainable=False)
    iter_cotrain_update = iter_cotrain.assign_add(1)
    return iter_cotrain, iter_cotrain_update

  def _save_state(self, saver, session, data, checkpts_path):
    """Saves the model and dataset state to files."""
    # Save variable state
    if checkpts_path:
      logging.info('Saving cotrain checkpoint at %s.', checkpts_path)
      saver.save(session, checkpts_path, write_meta_graph=False)

    # Save dataset state.
    if self.data_dir:
      logging.info('Saving self-labeled dataset backup.')
      data.save_state_to_file(self.data_dir)
