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

"""Training loop for adversarial training in multiple representation spaces."""

from absl import logging
import gin
from multi_representation_adversary import data
from multi_representation_adversary import helper
from multi_representation_adversary import resnet
from multi_representation_adversary import selectors
import tensorflow.compat.v2 as tf


@gin.configurable
def learning_rate_scheduler(epoch, values=(0.1, 0.01, 0.001),
                            breakpoints=(100, 150)):
  """Piecewise constant schedule for learning rate."""
  idx = sum(1 if epoch > b else 0 for b in breakpoints)
  return values[idx]


@gin.configurable
def train(ckpt_dir=None,
          summary_dir=None,
          epochs=200,
          steps_per_epoch=351,  # 45000 / 128 for CIFAR-10
          global_batch_size=128,
          model_fn=resnet.build_resnet_v1,
          lr_scheduler=learning_rate_scheduler,
          representation_list=(("identity", "none"),)):
  """Train a model with adversarial training in multiple representation spaces.

  Args:
    ckpt_dir: The directory to store model checkpoints.
    summary_dir: The directory to store training summaries.
    epochs: Maximum number of epochs to train for.
    steps_per_epoch: Number of training steps in each epoch.
    global_batch_size: Batch size across all processors/accelerators for each
      training step.
    model_fn: A callable which builds the model structure.
    lr_scheduler: A callable which returns the learning rate at any given epoch.
    representation_list: A list of (transform, attack) tuples representing the
      adversaries that this model should consider.
  """
  # Set up distributed training strategy first because all variables (model,
  # optimizer, etc) have to be created in the strategy's scope.
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    model = model_fn(return_logits=True)  # Other params are set in gin
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler(0),
                                        momentum=0.9)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def loss_fn(label, logit):
      # Normalize by global_batch_size, which is different from usual
      # (per-replica) batch size in a distributed training environment.
      return tf.nn.compute_average_loss(loss_obj(label, logit),
                                        global_batch_size=global_batch_size)

    metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy("loss",
                                                       from_logits=True),
        tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]

    # Compile a tf.function for training and eval (validation) steps for each
    # (transform, attack) tuple.
    representation_names = []
    train_step_fns, eval_step_fns = [], []
    for transform_name, attack_name in representation_list:
      representation_names.append(f"{transform_name}_{attack_name}")
      attack_fn = helper.build_attack_fn(model, transform_name, attack_name)
      train_step_fns.append(helper.build_train_step_fn(
          model, optimizer, loss_fn, metrics, attack_fn))
      eval_step_fns.append(helper.build_eval_step_fn(model, metrics, attack_fn))
    selector = selectors.construct_representation_selector(representation_names)

    # Create checkpoint object for saving model weights and selector state.
    checkpoint = tf.train.Checkpoint(model=model, selector=selector)
    ckpt_mgr = tf.train.CheckpointManager(checkpoint, ckpt_dir,
                                          max_to_keep=None)
    restored_path = ckpt_mgr.restore_or_initialize()
    if restored_path:
      logging.info("Restored checkpoint %s", restored_path)
      start_epoch = int(restored_path.rsplit("-", 1)[-1])  # path like "ckpt-N"
      total_steps = start_epoch * steps_per_epoch
    else:
      logging.info("Model initialized")
      start_epoch, total_steps = 0, 0
      ckpt_mgr.save(0)

    train_dataset = data.get_training_dataset(global_batch_size)
    valid_dataset = data.get_validation_dataset(global_batch_size)

    with tf.summary.create_file_writer(summary_dir).as_default():
      for epoch in range(start_epoch + 1, epochs + 1):
        logging.info("Epoch %d", epoch)

        # Learning rate decay
        if lr_scheduler(epoch) != optimizer.learning_rate:
          optimizer.learning_rate = lr_scheduler(epoch)
          logging.info("New learning rate: %g", optimizer.learning_rate)

        # Training
        dist_dataset = strategy.experimental_distribute_dataset(
            train_dataset.take(steps_per_epoch))
        for x, y in dist_dataset:
          selected_idx = selector.select(total_steps)
          train_step_fn = train_step_fns[selected_idx]
          per_replica_loss = strategy.run(train_step_fn, args=(x, y))
          loss_value = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                       per_replica_loss, axis=None)
          if total_steps % 50 == 0:
            tf.summary.scalar("train/batch_loss", loss_value, step=total_steps)
          total_steps += 1

        for metric in metrics:
          tf.summary.scalar(f"train/{metric.name}", metric.result(), step=epoch)
          metric.reset_states()

        # Maybe update the selector's state
        if selector.should_update(epoch):
          logging.info("Evaluate on validation set and update selector state")
          validation_losses = []
          dist_val_dataset = strategy.experimental_distribute_dataset(
              valid_dataset)
          for i, eval_step_fn in enumerate(eval_step_fns):
            for x, y in dist_val_dataset:
              strategy.run(eval_step_fn, args=(x, y))
            validation_losses.append(metrics[0].result())  # Crossentropy loss
            for metric in metrics:
              name = f"validation/{metric.name}/{representation_names[i]}"
              tf.summary.scalar(name, metric.result(), step=epoch)
              metric.reset_states()
          selector.update(epoch, validation_losses)

        # Save a checkpoint
        ckpt_mgr.save(epoch)
