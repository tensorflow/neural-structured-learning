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

"""Main function to start training / evaluation."""

import os

from absl import app
from absl import flags
import gin
from multi_representation_adversary import evaluator
from multi_representation_adversary import trainer
import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS

flags.DEFINE_multi_string("gin_configs", [],
                          "List of paths to the config files.")
flags.DEFINE_multi_string("gin_bindings", [],
                          "Newline separated list of Gin parameter bindings.")
flags.DEFINE_string("summary_dir", "", "Summary directory for Tensorboard.")
flags.DEFINE_string("ckpt_dir", "", "Checkpoint directory for saving models.")
flags.DEFINE_bool("run_train", False, "Whether to train a model.")
flags.DEFINE_bool("run_eval", False, "Whether to evaluate a model.")


def save_gin_config(config_str, path):
  """Saves the config to a file."""
  if tf.io.gfile.exists(path):
    # Add a suffix to avoid overwriting existing files.
    suffix = 2
    while tf.io.gfile.exists(f"{path}.{suffix}"):
      suffix += 1
    path += f".{suffix}"
  with tf.io.gfile.GFile(path, "w") as f:
    f.write(config_str)


def main(argv):
  del argv  # Unused.
  gin.parse_config_files_and_bindings(FLAGS.gin_configs, FLAGS.gin_bindings)

  tf.io.gfile.makedirs(FLAGS.summary_dir)
  save_gin_config(gin.config_str(),
                  os.path.join(FLAGS.summary_dir, "config.parsed.gin"))

  if FLAGS.run_train:
    trainer.train(ckpt_dir=FLAGS.ckpt_dir, summary_dir=FLAGS.summary_dir)
  if FLAGS.run_eval:
    evaluator.evaluate(ckpt_dir=FLAGS.ckpt_dir, summary_dir=FLAGS.summary_dir)

  save_gin_config(gin.operative_config_str(),
                  os.path.join(FLAGS.summary_dir, "config.operative.gin"))


if __name__ == "__main__":
  app.run(main)
