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

"""Abstract class for generating neighbors.

This abstract class will be inherited by classes with actual implementation for
generating neigbors (e.g., adversarial neighbors or graph neighbors).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class GenNeighbor(object):
  """Abstract class for generating neighbors.

  This class is to be inherited by the class that actually implements the method
  to generate neighbors.
  """

  def __init__(self):
    raise NotImplementedError

  def gen_neighbor(self):
    raise NotImplementedError
