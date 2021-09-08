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

"""Evaluator for adversarial training in multiple representation spaces."""

import functools
import hashlib
import os
import time

from absl import logging
import gin
from multi_representation_adversary import data
from multi_representation_adversary import helper
from multi_representation_adversary import resnet
import tensorflow.compat.v2 as tf


def _hash(tensor):
  content = tf.make_tensor_proto(tensor).SerializeToString()
  return hashlib.md5(content).hexdigest()


def _evaluate_dataset(dataset, eval_step_fn, strategy, prediction_writer=None):
  """Evaluates the dataset using the given step_fn and distribution strategy.

  Args:
    dataset: A `tf.data.Dataset` of the evaluation data. Each batch contains an
      image tensor and a label tensor.
    eval_step_fn: A callable which takes a batch of (image, label) and returns
      logits. This can have side effects like updating metric objects.
    strategy: A `tf.distribute.Strategy` for running on one or more devices.
    prediction_writer: A file-like object to save each image's prediction for
      further analysis (e.g. union attack).

  Returns:
    None.
  """
  dist_dataset = strategy.experimental_distribute_dataset(dataset)
  for step, (dist_images, dist_labels) in enumerate(dist_dataset):
    if step % 50 == 0:
      logging.info("Evaluation step %d", step)
    if strategy:
      dist_logits = strategy.run(eval_step_fn, args=(dist_images, dist_labels))
      logits = strategy.gather(dist_logits, axis=0)
      images = strategy.gather(dist_images, axis=0)
      labels = strategy.gather(dist_labels, axis=0)
    else:
      images, labels = dist_images, dist_labels
      logits = eval_step_fn(images, labels)
    if prediction_writer:
      image_hashes = map(_hash, images)
      predictions = tf.math.argmax(logits, axis=-1).numpy()
      for x_hash, pred, label in zip(image_hashes, predictions, labels.numpy()):
        prediction_writer.write(f"{x_hash}\t{pred}\t{label}\n")


def load_and_aggregate(ckpt_dir, aggregating_epochs, aggregation_method,
                       base_model_fn):
  """Returns an aggregated model from checkpoints at given epochs.

  Args:
    ckpt_dir: Checkpoint directory to load the models.
    aggregating_epochs: A list of epoch numbers which corresponding checkpoints
      are loaded and aggregated.
    aggregation_method: How to aggregate checkpoints. Set to "avg_weight" for
      averaging model weights, or "ensemble" for ensemble (averaging
      probablistic predicitons).
    base_model_fn: A callable which builds the model structure.

  Returns:
    A callable which can make inference on input.
  """
  if aggregation_method not in ("avg_weight", "ensemble"):
    raise ValueError(f"Unknown aggregation method: {aggregation_method}")
  models = []
  for epoch in aggregating_epochs:
    logging.info("Loading model at epoch %d", epoch)
    path = os.path.join(ckpt_dir, f"ckpt-{epoch}")
    models.append(helper.load_checkpoint(path, base_model_fn))

  logging.info("Aggregating models using %s", aggregation_method)
  if aggregation_method == "avg_weight":
    model_weights = [mdl.get_weights() for mdl in models]
    agg_weights = [sum(ws) / len(models) for ws in zip(*model_weights)]
    agg_model = base_model_fn()
    agg_model.set_weights(agg_weights)
    return agg_model
  else:  # ensemble
    @tf.function
    def ensemble_model(x):
      predictions = [tf.nn.softmax(model(x), axis=-1) for model in models]
      avg_pred = tf.reduce_mean(tf.stack(predictions, axis=0), axis=0)
      return tf.math.log(avg_pred)  # return in logit space
    return ensemble_model


@gin.configurable
def evaluate(ckpt_dir=None,
             summary_dir=None,
             epochs=200,
             batch_size=128,
             checkpoint_timeout=60 * 60 * 5,
             num_aggregate=1,
             aggregation_method="avg_weight",  # or "ensemble"
             aggregation_interval=1,
             model_fn=resnet.build_resnet_v1,
             representation_list=(("identity", "none"),),
             should_write_final_predictions=True):
  """Watches and evaluates model checkpoints.

  Args:
    ckpt_dir: The directory to store model checkpoints.
    summary_dir: The directory to store evaluation summaries.
    epochs: Maximum number of epochs.
    batch_size: The batch size for each evaluation step.
    checkpoint_timeout: Number of seconds to wait for a new checkpoint.
    num_aggregate: Number of checkpoints to aggregate.
    aggregation_method: How to aggregate checkpoints. Set to "avg_weight" for
      averaging model weights, or "ensemble" for ensemble (averaging
      probablistic predicitons).
    aggregation_interval: Number of epochs between two consecutive checkpoints
      for aggregation.
    model_fn: A callable which builds the model structure.
    representation_list: A list of (transform, attack) tuples representing the
      adversaries that this model should consider.
    should_write_final_predictions: Whether to write the last epoch's
      predictions of each test example to a file. This can be used to compute
      the accuracy under a "union" attack offline.
  """
  metric_prefix = "test"
  if num_aggregate > 1:
    metric_prefix += (
        f"_{aggregation_method}_{num_aggregate}_{aggregation_interval}")

  # Set up distributed training strategy first because all variables (model,
  # optimizer, etc) have to be created in the strategy's scope.
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    base_model_fn = functools.partial(model_fn, return_logits=True)
    metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy("loss",
                                                       from_logits=True),
        tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]
    test_dataset = data.get_test_dataset(batch_size)

    epoch = -1
    last_eval_time, last_eval_ckpt = time.time(), None
    while time.time() - last_eval_time < checkpoint_timeout and epoch < epochs:
      latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
      if not latest_checkpoint or latest_checkpoint == last_eval_ckpt:
        logging.info("No new checkpoints found. Sleeping for 10 seconds.")
        time.sleep(10)
        continue

      epoch = int(latest_checkpoint.rsplit("-", 1)[-1])  # path like "ckpt-NUM"
      aggregating_epochs = [
          epoch - i * aggregation_interval for i in range(num_aggregate)]
      if aggregating_epochs[-1] < 0:
        logging.info("Not enough checkpoints to aggregate. Sleep 10 seconds.")
        time.sleep(10)
        continue

      # Load the model
      if len(aggregating_epochs) > 1:
        model = load_and_aggregate(ckpt_dir, aggregating_epochs,
                                   aggregation_method, base_model_fn)
      else:
        model = helper.load_checkpoint(latest_checkpoint, base_model_fn)

      # Evaluate
      for transform_name, attack_name in representation_list:
        logging.info("Evaluating with %s & %s", transform_name, attack_name)
        attack_fn = helper.build_attack_fn(model, transform_name, attack_name)
        eval_step_fn = helper.build_eval_step_fn(model, metrics, attack_fn)
        representation_name = f"{transform_name}_{attack_name}"
        if epoch == epochs and should_write_final_predictions:
          # Save per-instance predictions for calculating union-attack accuracy.
          # This is much faster than evaluating against a union attack online.
          path = os.path.join(summary_dir, f"results_{representation_name}.tsv")
          if tf.io.gfile.exists(path) and tf.io.gfile.stat(path).length > 0:
            logging.info("Predictions found. No need to re-evaluate.")
            break
          with tf.io.gfile.GFile(path, "w") as prediction_writer:
            _evaluate_dataset(test_dataset, eval_step_fn, strategy,
                              prediction_writer)
        else:
          _evaluate_dataset(test_dataset, eval_step_fn, strategy)

        with tf.summary.create_file_writer(summary_dir).as_default():
          for metric in metrics:
            name = f"{metric_prefix}/{metric.name}/{representation_name}"
            tf.summary.scalar(name, metric.result(), step=epoch)
            metric.reset_states()
      last_eval_time, last_eval_ckpt = time.time(), latest_checkpoint
