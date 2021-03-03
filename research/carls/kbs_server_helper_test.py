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
"""Tests for kbs_server_helper's python cliff library."""

from research.carls import kbs_server_helper_pybind as kbs_server_helper

import tensorflow as tf


class KbsServerHelperTest(tf.test.TestCase):

  def testKnowledgeBankServiceOptions(self):
    options = kbs_server_helper.KnowledgeBankServiceOptions(True, 200, 10)
    self.assertTrue(options.run_locally)
    self.assertEqual(200, options.port)
    self.assertEqual(10, options.num_threads)

  def testKbsServerHelper(self):
    options = kbs_server_helper.KnowledgeBankServiceOptions(True, -1, 10)
    server = kbs_server_helper.KbsServerHelper(options)
    self.assertNotEqual('', server.address())
    server.Terminate()


if __name__ == '__main__':
  tf.test.main()
