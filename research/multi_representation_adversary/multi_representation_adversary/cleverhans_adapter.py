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
"""Adapter for CleverHans v3.1.0 to be run in TensorFlow 2.x environment.

The multi-representation adversary experiments are run in TensorFlow 2, but
depend on a version of the CleverHans package which expects TensorFlow 1. This
adapter glues them together by importing needed parts from CleverHans and
assigning their TensorFlow references to `tensorflow.compat.v1`.
"""

# pylint: disable=g-bad-import-order
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

# Expose the symbols used in function interfaces. This has to be done before
# actually importing CleverHans.
tf.GraphKeys = tfv1.GraphKeys

# pylint: disable=g-import-not-at-top
from cleverhans import compat
from cleverhans import utils_tf
from cleverhans.attacks import sparse_l1_descent
# pylint: enable=g-import-not-at-top
# pylint: enable=g-bad-import-order

# Bind the expected TensorFlow version.
compat.tf = tfv1
utils_tf.tf = tfv1
sparse_l1_descent.tf = tfv1
