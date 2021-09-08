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

"""Selection strategies for training with multiple adversarial representations.

A selector can select one representation for training at each step, and
maintain its internal state for subsequent selections. The state can also be
updated once every K epochs when the model is evaluated on the validation set.
"""

import gin
import tensorflow.compat.v2 as tf


class SelectionStrategy(tf.Module):
  """Base class for strategies to select representations.

  This base class handles sample and update frequencies, as well as logging
  selection statistics to TensorBoard. Sub-classes should implement _select()
  and _update().
  """

  def __init__(self, representation_names, sample_freq, update_freq):
    """Constructs a SelectionStrategy object.

    Args:
      representation_names: A list of representations names for tf.summary.
      sample_freq: Frequency to draw a new selection (in steps).
      update_freq: Frequency to update the selector's state (in epochs).
    """
    self.num_representations = len(representation_names)
    self.representation_names = representation_names
    self.sample_freq = sample_freq
    self.update_freq = update_freq
    # index of the selected representation
    self.current_selection = tf.Variable(0, trainable=False)
    self.last_selection_step = tf.Variable(-1, trainable=False)
    self.last_update_epoch = tf.Variable(0, trainable=False)
    self.selection_counter = tf.Variable([0] * self.num_representations)

  def select(self, step):
    """Returns the index of the selected representation for a training step."""
    if step - self.last_selection_step >= self.sample_freq:
      self.current_selection.assign(self._select())
      self.last_selection_step.assign(step)
    # Increment the counter for the newly selected item.
    self.selection_counter.scatter_add(
        tf.IndexedSlices(1, self.current_selection))
    return self.current_selection.numpy()

  def should_update(self, epoch):
    """Returns whether the strategy should update its state at this epoch."""
    return epoch - self.last_update_epoch >= self.update_freq

  def update(self, epoch, validation_losses):
    """Updates the strategy's state based on current validation losses.

    Args:
      epoch: Current epoch number.
      validation_losses: A list of numbers, one for each representation.
    """
    self._update(epoch, validation_losses)
    self.last_update_epoch.assign(epoch)
    # Log the counts since last update to the summary and reset the counts.
    for i in range(self.num_representations):
      tf.summary.scalar(
          f"representations/selected/{self.representation_names[i]}",
          self.selection_counter[i],
          step=epoch)
    self.selection_counter.assign([0] * self.num_representations)

  def _select(self):
    raise NotImplementedError

  def _update(self, epoch, validation_losses):
    """See update()."""
    raise NotImplementedError


class GreedyStrategy(SelectionStrategy):
  """Greedy strategy which selects the one with the highest validation loss."""

  def _select(self):
    # No needs to reselect since this strategy is deterministic.
    return self.current_selection.numpy()

  def _update(self, epoch, validation_losses):
    del epoch  # unused
    self.current_selection.assign(
        tf.cast(tf.argmax(validation_losses), self.current_selection.dtype))


class RoundRobinStrategy(SelectionStrategy):
  """Round-robin strategy which selects each representation sequentially."""

  def _select(self):
    return (self.current_selection + 1) % self.num_representations

  def _update(self, epoch, validation_losses):
    pass


@gin.configurable
def eta_scheduler(epoch, values=(0.1,), breakpoints=()):
  """Piecewise constant schedule for eta (selector weight learning rate)."""
  idx = sum(1 if epoch > b else 0 for b in breakpoints)
  return values[idx]


class MultiplicativeWeightStrategy(SelectionStrategy):
  """Multiplicative-weight strategy which samples representations adaptively."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Weights of each representation. Each selection is a sample drawn
    # proportionally to the weights.
    # TODO(csferng): Store the weights in logit space.
    self.weights = tf.Variable(tf.ones(self.num_representations))
    self.current_selection.assign(self._select())

  def _select(self):
    logits = tf.math.log(self.weights / tf.reduce_sum(self.weights))
    return tf.random.categorical(tf.reshape(logits, [1, -1]), 1)[0][0].numpy()

  def _update(self, epoch, validation_losses):
    validation_losses = tf.convert_to_tensor(validation_losses)
    eta = eta_scheduler(epoch)
    self.weights.assign(self.weights * tf.math.exp(eta * validation_losses))
    for i in range(self.num_representations):
      tf.summary.scalar(
          f"representations/weight/{self.representation_names[i]}",
          self.weights[i],
          step=epoch)


STRATEGY_CLASSES = {
    "greedy": GreedyStrategy,
    "roundrobin": RoundRobinStrategy,
    "multiweight": MultiplicativeWeightStrategy,
}


@gin.configurable
def construct_representation_selector(representation_names,
                                      selection_strategy="multiweight",
                                      sample_freq=351,  # in number of steps
                                      update_freq=5):  # in number of epochs
  return STRATEGY_CLASSES[selection_strategy](representation_names, sample_freq,
                                              update_freq)
