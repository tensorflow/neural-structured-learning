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
"""Main logic for training the A2N model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import math
import os

from absl import app
from absl import flags
from absl import logging
import clueweb_text_graph
import dataset
import graph
import losses
import metrics
import models
import numpy as np
import slim
from tensorboard.plugins import projector
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache
import text_graph
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string("kg_file", None, "path to kg file")
flags.DEFINE_string("output_dir", None, "output dir for summaries/logs")
flags.DEFINE_string("dev_kg_file", None, "path to dev kg file")
flags.DEFINE_string("test_kg_file", None, "path to test kg file")
flags.DEFINE_string("model_path", None, "path to model if testing only")
flags.DEFINE_boolean("evaluate", False, "run eval loop")
flags.DEFINE_boolean("test_only", False, "if test only")
flags.DEFINE_integer("global_step", None,
                     "global_step to restore model for testing")
flags.DEFINE_integer("num_epochs", 5, "number of train epochs")
flags.DEFINE_integer("batchsize", 64, "batchsize for training")
flags.DEFINE_integer("test_batchsize", 10, "batchsize for testing")
flags.DEFINE_integer("max_neighbors", None,
                     "maximum neighbors to use during training")
flags.DEFINE_integer("max_negatives", None,
                     "maximum number of negative entities to sample")
flags.DEFINE_integer("emb_dim", 100,
                     "dimension of entity and relation embeddings")
flags.DEFINE_float("entity_encoder_dropout", 1.0,
                   "dropout for entity embeddings")
flags.DEFINE_float("relation_encoder_dropout", 1.0,
                   "dropout for relation embeddings")
flags.DEFINE_float("init_entity_encoder_dropout", 1.0,
                   "dropout for init entity embeddings in attention")
flags.DEFINE_float("attention_encoder_dropout", 1.0,
                   "dropout for attention encoder")
flags.DEFINE_boolean("use_separate_attention_emb", False,
                     "use separate entity embeddings for computing attention")
flags.DEFINE_integer("num_parallel_preprocess", 64,
                     "number of processes to use in dataset preprocessing")
flags.DEFINE_integer("prefetch_examples", 10, "number of examples to prefetch")
flags.DEFINE_integer("shuffle_buffer", 50000,
                     "buffer for shuffling training examples")
flags.DEFINE_float("learning_rate", 0.001, "learning for optimizer")
flags.DEFINE_float("grad_clip", None, "Clip gradient norm during training")
flags.DEFINE_integer("save_every", 100, "save model every this many steps")
flags.DEFINE_string("entity_names_file", None,
                    "mapping of Freebase mid to names")
flags.DEFINE_enum("model", "attention",
                  ["distmult", "attention", "source_attention",
                   "source_rel_attention", "source_path_attention"],
                  "the model to use")
flags.DEFINE_bool("use_tanh", False, "use tanh non-linearity on embeddings")
flags.DEFINE_enum("attention_type", "bilinear",
                  ["bilinear", "cosine", "sigmoid_bilinear",
                   "sigmoid_avg_bilinear", "relation"],
                  "type of attention to use for attention model")
flags.DEFINE_bool("analyze", False, "analyze model")
flags.DEFINE_integer("max_path_length", None,
                     "maximum path length for path attention models")
flags.DEFINE_string("text_kg_file", None, "path to text data")
flags.DEFINE_integer("max_text_len", None, "max length of text")
flags.DEFINE_integer("max_vocab_size", None, "max number of text words")
flags.DEFINE_integer("min_word_freq", None, "min freq threshold for text words")
flags.DEFINE_integer("max_text_neighbors", None, "max text neighbors")
flags.DEFINE_float("text_encoder_dropout", 1.0, "dropout for text cnn")
flags.DEFINE_list("text_encoder_filter_widths", ["3", "5", "7"],
                  "filter widths for cnn")
flags.DEFINE_enum("text_encoder_nonlinearity", "tanh", ["relu", "tanh"],
                  "non-linearity to use for TextCNN")
flags.DEFINE_integer("text_encoder_num_filters", 64, "num filters for cnn")

flags.DEFINE_string("clueweb_sentences", None,
                    "path to clueweb sentences (or data formatted like cw)")
flags.DEFINE_string("clueweb_data", None,
                    "path to clueweb data (or data formatted like cw)")
flags.DEFINE_string("clueweb_embeddings", None,
                    "path to clueweb embeddings (or data formatted like cw)")
flags.DEFINE_integer("text_emb_dim", None, "embedding dim for clueweb text")
flags.DEFINE_integer("subsample_text_rels", None,
                     "subsample text to max this many per pair")

flags.DEFINE_string("master", "local",
                    """BNS name of the TensorFlow master to use.""")
flags.DEFINE_integer("task", 0,
                     """Task id of the replica running the training.""")
flags.DEFINE_integer("ps_tasks", 0, """Number of tasks in the ps job.
                            If 0 no ps job is used.""")

flags.mark_flag_as_required("kg_file")
flags.mark_flag_as_required("output_dir")


def add_embedding_to_projector(projector_config, emb_name, emb_metadata_path):
  embedding_conf = projector_config.embeddings.add()
  embedding_conf.tensor_name = emb_name
  embedding_conf.metadata_path = emb_metadata_path


def get_train_op(loss, optimizer, grad_clip=None, global_step=None):
  """Make a train_op apply gradients to loss using optimizer.

  Args:
   loss: the loss function to optimize
   optimizer: the optimizer to compute and apply gradients
   grad_clip: clip gradient norms by the value supplied (default dont clip)
   global_step: tf.placeholder for global_step

  Returns:
   train_op: the training op to run
   grads_and_vars: the gradients and variables for debugging
   var_names: the variable names for debugging
   capped_grads_and_vars: for debugging
  """
  variables = tf.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(loss, variables)
  var_names = [v.name for v in variables]
  logging.info("Trainable variables:")
  for var in var_names:
    logging.info("\t %s", var)
  logging.debug(grads_and_vars)
  grad_var_norms = [(tf.global_norm([gv[1]]), tf.global_norm([gv[0]]))
                    for gv in grads_and_vars]

  if grad_clip:
    capped_grads_and_vars = [(tf.clip_by_norm(gv[0], grad_clip), gv[1])
                             for gv in grads_and_vars]
  else:
    capped_grads_and_vars = grads_and_vars
  # norms of gradients for debugging
  # grad_norms = [tf.sqrt(tf.reduce_sum(tf.square(grad)))
  #               for grad, _ in grads_and_vars]
  train_op = optimizer.apply_gradients(capped_grads_and_vars,
                                       global_step=global_step)
  return train_op, grad_var_norms, var_names, capped_grads_and_vars


def read_graph_data(
    kg_file, add_reverse_graph, add_inverse_edge, mode,
    num_epochs, batchsize, max_neighbors, max_negatives,
    train_graph=None, text_kg_file=None, val_graph=None
):
  """Read graph, create dataset and build model."""
  # Read graphs and create datasets
  entity_vocab = relation_vocab = None
  if train_graph:
    entity_vocab = train_graph.entity_vocab
    relation_vocab = train_graph.relation_vocab
  if FLAGS.clueweb_data and mode == "train":
    graph_type = clueweb_text_graph.CWTextGraph
    text_kg_file = FLAGS.clueweb_data
  elif text_kg_file and mode == "train":
    graph_type = text_graph.TextGraph
    text_kg_file = FLAGS.text_kg_file
  else:
    graph_type = graph.Graph
    text_kg_file = None
  k_graph = graph_type(
      text_kg_file=text_kg_file,
      skip_new=True,
      max_text_len=FLAGS.max_text_len,
      max_vocab_size=FLAGS.max_vocab_size,
      min_word_freq=FLAGS.min_word_freq,
      kg_file=kg_file,
      add_reverse_graph=add_reverse_graph,
      add_inverse_edge=add_inverse_edge, mode=mode,
      entity_vocab=entity_vocab, relation_vocab=relation_vocab,
      max_path_length=FLAGS.max_path_length if mode == "train" else None,
      embeddings_file=FLAGS.clueweb_embeddings,
      sentence_vocab_file=FLAGS.clueweb_sentences,
      subsample=FLAGS.subsample_text_rels
  )
  if FLAGS.text_kg_file:
    max_text_len = FLAGS.max_text_len
    if mode == "train":
      max_text_len = max_text_len or k_graph.max_text_len
    elif train_graph:
      max_text_len = max_text_len or train_graph.max_text_len
  else:
    max_text_len = None
  k_data = dataset.Dataset(data_graph=k_graph, train_graph=train_graph,
                           mode=mode, num_epochs=num_epochs,
                           batchsize=batchsize,
                           max_neighbors=max_neighbors,
                           max_negatives=max_negatives,
                           model_type=FLAGS.model,
                           max_text_len=max_text_len,
                           max_text_neighbors=FLAGS.max_text_neighbors,
                           val_graph=val_graph)
  # Create the training data iterator and return the input tensors
  # with tf.device("/job:worker"):
  k_data.create_dataset_iterator(
      num_parallel=FLAGS.num_parallel_preprocess,
      prefetch=FLAGS.prefetch_examples,
      shuffle_buffer=FLAGS.shuffle_buffer
      # , device="worker" if FLAGS.master != "local" else "cpu"
  )

  return k_graph, k_data


def create_model(train_graph, iterator):
  """Create model and placeholders."""
  if FLAGS.clueweb_data:
    s, nbrs_s, text_nbrs_s, r, candidates, nbrs_candidates, labels, text_nbrs_s_emb = iterator.get_next()
  elif FLAGS.text_kg_file:
    s, nbrs_s, text_nbrs_s, r, candidates, nbrs_candidates, labels = \
      iterator.get_next()
  else:
    s, nbrs_s, r, candidates, nbrs_candidates, labels = iterator.get_next()

  # Create the attention model, this returns candidates scores and the model
  # encoders in a dict for creating feed_dict for all encoders
  is_train_ph = tf.placeholder_with_default(True, shape=[],
                                            name="is_train_ph")
  if FLAGS.model == "attention":
    with tf.variable_scope("attention_model", reuse=False):
      candidate_scores, model = models.attention_kbc_model(
          FLAGS, train_graph, is_train_ph,
          (s, nbrs_s, r, candidates, nbrs_candidates)
      )
  elif FLAGS.model == "source_attention":
    with tf.variable_scope("s_attention_model", reuse=False):
      candidate_scores, model = models.source_attention_kbc_model(
          FLAGS, train_graph, is_train_ph,
          (s, nbrs_s, r, candidates)
      )
  elif FLAGS.model in ["source_rel_attention", "source_path_attention"]:
    if FLAGS.clueweb_data:
      input_tensors = (s, nbrs_s, text_nbrs_s, text_nbrs_s_emb, r, candidates)
    elif FLAGS.text_kg_file:
      input_tensors = (s, nbrs_s, text_nbrs_s, r, candidates)
    else:
      input_tensors = (s, nbrs_s, r, candidates)
    with tf.variable_scope("s_attention_model", reuse=False):
      candidate_scores, model = models.source_attention_kbc_model(
          FLAGS, train_graph, is_train_ph,
          input_tensors, model_type=FLAGS.model
      )
  elif FLAGS.model == "distmult":
    with tf.variable_scope("distmult_model", reuse=False):
      candidate_scores, model = models.distmult_kbc_model(
          FLAGS, train_graph, is_train_ph,
          (s, r, candidates)
      )
  if FLAGS.clueweb_data:
    inputs = (s, nbrs_s, text_nbrs_s, text_nbrs_s_emb,
              r, candidates, nbrs_candidates)
  elif FLAGS.text_kg_file:
    inputs = (s, nbrs_s, text_nbrs_s, r, candidates, nbrs_candidates)
  else:
    inputs = (s, nbrs_s, r, candidates, nbrs_candidates)

  return candidate_scores, candidates, labels, model, is_train_ph, inputs


def evaluate():
  """Run evaluation on dev or test data."""
  add_inverse_edge = FLAGS.model in \
                     ["source_rel_attention", "source_path_attention"]
  if FLAGS.clueweb_data:
    train_graph = clueweb_text_graph.CWTextGraph(
        text_kg_file=FLAGS.clueweb_data,
        embeddings_file=FLAGS.clueweb_embeddings,
        sentence_vocab_file=FLAGS.clueweb_sentences,
        skip_new=True,
        kg_file=FLAGS.kg_file,
        add_reverse_graph=not add_inverse_edge,
        add_inverse_edge=add_inverse_edge,
        subsample=FLAGS.subsample_text_rels
    )
  elif FLAGS.text_kg_file:
    train_graph = text_graph.TextGraph(
        text_kg_file=FLAGS.text_kg_file,
        skip_new=True,
        max_text_len=FLAGS.max_text_len,
        max_vocab_size=FLAGS.max_vocab_size,
        min_word_freq=FLAGS.min_word_freq,
        kg_file=FLAGS.kg_file,
        add_reverse_graph=not add_inverse_edge,
        add_inverse_edge=add_inverse_edge,
        max_path_length=FLAGS.max_path_length
    )
  else:
    train_graph = graph.Graph(
        kg_file=FLAGS.kg_file,
        add_reverse_graph=not add_inverse_edge,
        add_inverse_edge=add_inverse_edge,
        max_path_length=FLAGS.max_path_length
    )
  # train_graph, _ = read_graph_data(
  #     kg_file=FLAGS.kg_file,
  #     add_reverse_graph=(FLAGS.model != "source_rel_attention"),
  #     add_inverse_edge=(FLAGS.model == "source_rel_attention"),
  #     mode="train", num_epochs=FLAGS.num_epochs, batchsize=FLAGS.batchsize,
  #     max_neighbors=FLAGS.max_neighbors,
  #     max_negatives=FLAGS.max_negatives
  # )
  val_graph = None
  if FLAGS.dev_kg_file:
    val_graph, eval_data = read_graph_data(
        kg_file=FLAGS.dev_kg_file,
        add_reverse_graph=not add_inverse_edge,
        add_inverse_edge=add_inverse_edge,
        # add_reverse_graph=False,
        # add_inverse_edge=False,
        mode="dev", num_epochs=1, batchsize=FLAGS.test_batchsize,
        max_neighbors=FLAGS.max_neighbors,
        max_negatives=FLAGS.max_negatives, train_graph=train_graph,
        text_kg_file=FLAGS.text_kg_file
    )
  if FLAGS.test_kg_file:
    _, eval_data = read_graph_data(
        kg_file=FLAGS.test_kg_file,
        add_reverse_graph=not add_inverse_edge,
        add_inverse_edge=add_inverse_edge,
        # add_reverse_graph=False,
        # add_inverse_edge=False,
        mode="test", num_epochs=1, batchsize=FLAGS.test_batchsize,
        max_neighbors=FLAGS.max_neighbors,
        max_negatives=None, train_graph=train_graph,
        text_kg_file=FLAGS.text_kg_file,
        val_graph=val_graph
    )
  if not FLAGS.dev_kg_file and not FLAGS.test_kg_file:
    raise ValueError("Evalution without a dev or test file!")

  iterator = eval_data.dataset.make_initializable_iterator()
  candidate_scores, candidates, labels, model, is_train_ph, inputs = \
    create_model(train_graph, iterator)

  # Create eval metrics
  # if FLAGS.dev_kg_file:
  batch_rr = metrics.mrr(candidate_scores, candidates, labels)
  mrr, mrr_update = tf.metrics.mean(batch_rr)
  mrr_summary = tf.summary.scalar("MRR", mrr)

  all_hits, all_hits_update, all_hits_summaries = [], [], []
  for k in [1, 3, 10]:
    batch_hits = metrics.hits_at_k(candidate_scores, candidates, labels, k=k)
    hits, hits_update = tf.metrics.mean(batch_hits)
    hits_summary = tf.summary.scalar("Hits_at_%d" % k, hits)
    all_hits.append(hits)
    all_hits_update.append(hits_update)
    all_hits_summaries.append(hits_summary)
  hits = tf.group(*all_hits)
  hits_update = tf.group(*all_hits_update)

  global_step = tf.Variable(0, name="global_step", trainable=False)
  current_step = tf.Variable(0, name="current_step", trainable=False,
                             collections=[tf.GraphKeys.LOCAL_VARIABLES])
  incr_current_step = tf.assign_add(current_step, 1)
  reset_current_step = tf.assign(current_step, 0)

  slim.get_or_create_global_step(graph=tf.get_default_graph())

  # best_hits = tf.Variable(0., trainable=False)
  # best_step = tf.Variable(0, trainable=False)
  # with tf.control_dependencies([hits]):
  #   update_best_hits = tf.cond(tf.greater(hits, best_hits),
  #                              lambda: tf.assign(best_hits, hits),
  #                              lambda: 0.)
  #   update_best_step = tf.cond(tf.greater(hits, best_hits),
  #                              lambda: tf.assign(best_step, global_step),
  #                              lambda: 0)
  # best_hits_summary = tf.summary.scalar("Best Hits@10", best_hits)
  # best_step_summary = tf.summary.scalar("Best Step", best_step)

  nexamples = eval_data.data_graph.tuple_store.shape[0]
  if eval_data.data_graph.add_reverse_graph:
    nexamples *= 2
  num_batches = math.ceil(nexamples / float(FLAGS.test_batchsize))
  local_init_op = tf.local_variables_initializer()

  if FLAGS.analyze:
    entity_names = utils.read_entity_name_mapping(FLAGS.entity_names_file)
    session = tf.Session()
    # summary_writer = tf.summary.FileWriter(FLAGS.output_dir, session.graph)
    init_op = tf.global_variables_initializer()
    session.run(init_op)
    session.run(local_init_op)
    saver = tf.train.Saver(tf.trainable_variables())
    ckpt_path = FLAGS.model_path + "/model.ckpt-%d" % FLAGS.global_step
    attention_probs = model["attention_encoder"].get_from_collection(
        "attention_probs"
    )
    if FLAGS.clueweb_data:
      s, nbrs_s, text_nbrs_s, text_nbrs_s_emb, r, candidates, _ = inputs
    elif FLAGS.text_kg_file:
      s, nbrs_s, text_nbrs_s, r, candidates, _ = inputs
    else:
      s, nbrs_s, r, candidates, _ = inputs
    saver.restore(session, ckpt_path)
    session.run(iterator.initializer)
    num_attention = 5
    nsteps = 0
    outf_correct = open(FLAGS.output_dir + "/analyze_correct.txt", "w+")
    outf_incorrect = open(
        FLAGS.output_dir + "/analyze_incorrect.txt", "w+"
    )
    ncorrect = 0
    analyze_outputs = [candidate_scores, s, nbrs_s, r, candidates, labels,
                       attention_probs]
    if FLAGS.text_kg_file:
      analyze_outputs.append(text_nbrs_s)
    while True:
      try:
        analyze_vals = session.run(analyze_outputs, {is_train_ph: False})
        if FLAGS.text_kg_file:
          cscores, se, nbrs, qr, cands, te, nbr_attention_probs, text_nbrs = \
            analyze_vals
        else:
          cscores, se, nbrs, qr, cands, te, nbr_attention_probs = analyze_vals
        # import pdb; pdb.set_trace()
        pred_ids = cscores.argmax(1)
        for i in range(se.shape[0]):
          sname = train_graph.inverse_entity_vocab[se[i]]
          if sname in entity_names:
            sname = entity_names[sname]
          rname = train_graph.inverse_relation_vocab[qr[i]]
          pred_target = cands[i, pred_ids[i]]
          pred_name = train_graph.inverse_entity_vocab[pred_target]
          if pred_name in entity_names:
            pred_name = entity_names[pred_name]
          tname = train_graph.inverse_entity_vocab[te[i][0]]
          if tname in entity_names:
            tname = entity_names[tname]
          if te[i][0] == pred_target:
            outf = outf_correct
            ncorrect += 1
          else:
            outf = outf_incorrect
          outf.write("\n(%d) %s, %s, ? \t Pred: %s \t Target: %s" %
                     (nsteps+i+1, sname, rname, pred_name, tname))
          top_nbrs_index = np.argsort(nbr_attention_probs[i, :])[::-1]
          outf.write("\nTop Nbrs:")
          for j in range(num_attention):
            nbr_index = top_nbrs_index[j]
            if nbr_index < FLAGS.max_neighbors:
              nbr_id = nbrs[i, nbr_index, :]
              nbr_name = ""
              for k in range(0, nbrs.shape[-1], 2):
                ent_name = train_graph.inverse_entity_vocab[nbr_id[k+1]]
                if ent_name in entity_names:
                  ent_name = entity_names[ent_name]
                rel_name = train_graph.inverse_relation_vocab[nbr_id[k]]
                nbr_name += "(%s, %s)" % (rel_name, ent_name)
            else:
              # Text Relation
              text_nbr_ids = text_nbrs[i, nbr_index - FLAGS.max_neighbors, :]
              text_nbr_ent = text_nbr_ids[0]
              ent_name = train_graph.inverse_entity_vocab[text_nbr_ent]
              if ent_name in entity_names:
                ent_name = entity_names[ent_name]
              rel_name = train_graph.get_relation_text(text_nbr_ids[1:])
              nbr_name = "(%s, %s)" % (rel_name, ent_name)
            outf.write("\n\t\t %s Prob: %.4f" %
                       (nbr_name, nbr_attention_probs[i, nbr_index]))
        nsteps += se.shape[0]
        tf.logging.info("Current hits@1: %.3f", ncorrect * 1.0 / (nsteps))

      except tf.errors.OutOfRangeError:
        break
    outf_correct.close()
    outf_incorrect.close()
    return

  class DataInitHook(tf.train.SessionRunHook):

    def after_create_session(self, sess, coord):
      sess.run(iterator.initializer)
      sess.run(reset_current_step)

  if FLAGS.test_only:
    ckpt_path = FLAGS.model_path + "/model.ckpt-%d" % FLAGS.global_step
    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=ckpt_path,
        logdir=FLAGS.output_dir,
        variables_to_restore=tf.trainable_variables() + [global_step],
        initial_op=tf.group(local_init_op, iterator.initializer),
        # initial_op=iterator.initializer,
        num_evals=num_batches,
        eval_op=tf.group(mrr_update, hits_update, incr_current_step),
        eval_op_feed_dict={is_train_ph: False},
        final_op=tf.group(mrr, hits),
        final_op_feed_dict={is_train_ph: False},
        summary_op=tf.summary.merge([mrr_summary]+ all_hits_summaries),
        hooks=[DataInitHook(),
               tf.train.LoggingTensorHook(
                   {"mrr": mrr, "hits": hits, "step": current_step},
                   every_n_iter=1
               )]
    )
  else:
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.model_path,
        logdir=FLAGS.output_dir,
        variables_to_restore=tf.trainable_variables() + [global_step],
        initial_op=tf.group(local_init_op, iterator.initializer),
        # initial_op=iterator.initializer,
        num_evals=num_batches,
        eval_op=tf.group(mrr_update, hits_update, incr_current_step),
        eval_op_feed_dict={is_train_ph: False},
        final_op=tf.group(mrr, hits),
        final_op_feed_dict={is_train_ph: False},
        summary_op=tf.summary.merge([mrr_summary] +  all_hits_summaries),
        max_number_of_evaluations=None,
        eval_interval_secs=60,
        hooks=[DataInitHook(),
               tf.train.LoggingTensorHook(
                   {"mrr": mrr, "hits": hits, "step": current_step},
                   every_n_iter=1
               )]
    )


def train():
  """Running the main training loop with given parameters."""
  if FLAGS.task == 0 and not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # Read train/dev/test graphs, create datasets and model
  add_inverse_edge = FLAGS.model in \
                     ["source_rel_attention", "source_path_attention"]
  train_graph, train_data = read_graph_data(
      kg_file=FLAGS.kg_file,
      add_reverse_graph=not add_inverse_edge,
      add_inverse_edge=add_inverse_edge,
      mode="train",
      num_epochs=FLAGS.num_epochs, batchsize=FLAGS.batchsize,
      max_neighbors=FLAGS.max_neighbors,
      max_negatives=FLAGS.max_negatives,
      text_kg_file=FLAGS.text_kg_file
  )

  worker_device = "/job:{}".format(FLAGS.brain_job_name)
  with tf.device(
      tf.train.replica_device_setter(
          FLAGS.ps_tasks, worker_device=worker_device)):
    iterator = train_data.dataset.make_one_shot_iterator()
    candidate_scores, _, labels, model, is_train_ph, _ = create_model(
        train_graph, iterator
    )

  # Create train loss and training op
  loss = losses.softmax_crossentropy(logits=candidate_scores, labels=labels)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  global_step = tf.Variable(0, name="global_step", trainable=False)
  train_op = get_train_op(loss, optimizer, FLAGS.grad_clip,
                          global_step=global_step)
  tf.summary.scalar("Loss", loss)

  run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
  session_config = tf.ConfigProto(log_device_placement=True)

  # Create tf training session
  scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1000))
  # ckpt_hook = tf.train.CheckpointSaverHook(
  #     checkpoint_dir=FLAGS.output_dir, scaffold=scaffold,
  #     save_steps=FLAGS.save_every
  # )
  # summary_hook = tf.train.SummarySaverHook(
  #     save_secs=60, output_dir=FLAGS.output_dir,
  #     summary_op=tf.summary.merge_all()
  # )
  session = tf.train.MonitoredTrainingSession(
      master=FLAGS.master,
      is_chief=(FLAGS.task == 0),
      checkpoint_dir=FLAGS.output_dir,
      save_checkpoint_steps=FLAGS.save_every,
      scaffold=scaffold,
      save_summaries_secs=60,
      # hooks=[summary_hook],
      # chief_only_hooks=[ckpt_hook],
      config=session_config
  )

  # Create embeddings visualization
  if FLAGS.task == 0:
    utils.save_embedding_vocabs(FLAGS.output_dir, train_graph,
                                FLAGS.entity_names_file)
    pconfig = projector.ProjectorConfig()
    add_embedding_to_projector(
        pconfig, model["entity_encoder"].embeddings.name.split(":")[0],
        os.path.join(FLAGS.output_dir, "entity_vocab.tsv")
    )
    add_embedding_to_projector(
        pconfig, model["relation_encoder"].embeddings.name.split(":")[0],
        os.path.join(FLAGS.output_dir, "relation_vocab.tsv")
    )
    if FLAGS.text_kg_file:
      word_embeddings = model["text_encoder"].word_embedding_encoder.embeddings
      add_embedding_to_projector(
          pconfig, word_embeddings.name.split(":")[0],
          os.path.join(FLAGS.output_dir, "word_vocab.tsv")
      )
    projector.visualize_embeddings(
        SummaryWriterCache.get(FLAGS.output_dir), pconfig
    )

  # Main training loop
  running_total_loss = 0.
  nsteps = 0
  gc.collect()
  while True:
    try:
      current_loss, _, _ = session.run(
          [loss, train_op, global_step],
          # feed_dict={is_train_ph: True, handle: train_iterator_handle},
          feed_dict={is_train_ph: True},
          options=run_options
      )
      nsteps += 1
      running_total_loss += current_loss
      tf.logging.info("Step %d, loss: %.3f, running avg loss: %.3f",
                      nsteps, current_loss, running_total_loss / nsteps)
      if nsteps %2 == 0:
        gc.collect()
    except tf.errors.OutOfRangeError:
      tf.logging.info("End of Traning Epochs after %d steps", nsteps)
      break


def main(argv):
  del argv
  if FLAGS.test_only or FLAGS.evaluate or FLAGS.analyze:
    evaluate()
  else:
    train()


if __name__ == "__main__":
  app.run(main)
