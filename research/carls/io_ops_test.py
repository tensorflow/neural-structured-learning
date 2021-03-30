# Copyright 2021 Google LLC
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

from absl import flags
from research.carls import context
from research.carls import dynamic_embedding_ops as de_ops
from research.carls import io_ops
from research.carls.testing import test_util
import tensorflow as tf

FLAGS = flags.FLAGS


class IoOpsTest(tf.test.TestCase):

  def setUp(self):
    super(IoOpsTest, self).setUp()
    self._config = test_util.default_de_config(2)
    self._service_server = test_util.start_kbs_server()
    self._kbs_address = 'localhost:%d' % self._service_server.port()
    context.clear_all_collection()

  def tearDown(self):
    self._service_server.Terminate()
    super(IoOpsTest, self).tearDown()

  def test_save_knowledge_bank(self):
    # Adds an embedding with values [4, 5].
    pattern1 = (
        FLAGS.test_tmpdir + '/knowledge_bank_data_[0-9]+_[0-9]+_[0-9]+' +
        '/emb1/embedding_store_meta_data.pbtxt')
    de_ops.dynamic_embedding_update(['first'],
                                    tf.constant([4.0, 5.0]),
                                    self._config,
                                    'emb1',
                                    service_address=self._kbs_address)
    saved_paths = io_ops.save_knowledge_bank(FLAGS.test_tmpdir,
                                             self._kbs_address)
    self.assertLen(saved_paths, 1)
    self.assertRegex(saved_paths[0].numpy()[0].decode(), pattern1)

    # Add another embedding data.
    pattern2 = (
        FLAGS.test_tmpdir + '/knowledge_bank_data_[0-9]+_[0-9]+_[0-9]+' +
        '/emb2/embedding_store_meta_data.pbtxt')
    de_ops.dynamic_embedding_update(['first'],
                                    tf.constant([5.0, 6.0]),
                                    self._config,
                                    'emb2',
                                    service_address=self._kbs_address)
    saved_paths = io_ops.save_knowledge_bank(FLAGS.test_tmpdir,
                                             self._kbs_address)
    self.assertLen(saved_paths, 2)
    self.assertRegex(saved_paths[0].numpy()[0].decode(), pattern1)
    self.assertRegex(saved_paths[1].numpy()[0].decode(), pattern2)

    # Only save selected embedding.
    new_saved_paths = io_ops.save_knowledge_bank(
        FLAGS.test_tmpdir, self._kbs_address, var_names=['emb2'])
    self.assertLen(new_saved_paths, 1)
    self.assertRegex(new_saved_paths[0].numpy()[0].decode(), pattern2)
    self.assertNotEqual(new_saved_paths[0].numpy()[0],
                        saved_paths[0].numpy()[0])

  def test_restore_knowledge_bank(self):
    de_ops.dynamic_embedding_update(['first'],
                                    tf.constant([4.0, 5.0]),
                                    self._config,
                                    'emb',
                                    service_address=self._kbs_address)
    saved_paths = io_ops.save_knowledge_bank(FLAGS.test_tmpdir,
                                             self._kbs_address)
    self.assertLen(saved_paths, 1)

    # Now updates the embedding value.
    de_ops.dynamic_embedding_update(['first'],
                                    tf.constant([10.0, 20.0]),
                                    self._config,
                                    'emb',
                                    service_address=self._kbs_address)

    # Checks it is updated.
    embedding = de_ops.dynamic_embedding_lookup(
        ['first'], self._config, 'emb', service_address=self._kbs_address)
    self.assertAllClose(embedding.numpy(), [[10.0, 20.0]])

    # Now restore the knowledge bank.
    io_ops.restore_knowledge_bank(
        self._config,
        'emb',
        saved_paths[0].numpy()[0],
        service_address=self._kbs_address)

    # Checks it is restored.
    embedding = de_ops.dynamic_embedding_lookup(
        ['first'], self._config, 'emb', service_address=self._kbs_address)
    self.assertAllClose(embedding.numpy(), [[4.0, 5.0]])


if __name__ == '__main__':
  tf.test.main()
