# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Tool to generate api_docs for neural_structured_learning.

# How to run

Install tensorflow_docs if needed:

```
pip install git+https://github.com/tensorflow/docs
```

Run the docs generator:

```shell
python build_docs.py \
--output_dir=/tmp/neural_structured_learning_api
```

Note:
  If duplicate or spurious docs are generated, consider excluding them via the
  `private_map` argument to `generate_lib.DocGenerator()` below. Or
  `api_generator.doc_controls`
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import neural_structured_learning as nsl

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

_OUTPUT_DIR = flags.DEFINE_string("output_dir",
                                  "/tmp/neural_structured_learning_api",
                                  "Where to output the docs")
_CODE_URL_PREFIX = flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/tensorflow/neural-structured-learning/blob/master/neural_structured_learning",
    "The url prefix for links to code.")
_SEARCH_HINTS = flags.DEFINE_bool(
    "search_hints", True,
    "Include metadata search hints in the generated files")
_SITE_PATH = flags.DEFINE_string("site_path",
                                 "neural_structured_learning/api_docs/python",
                                 "Path prefix in the _toc.yaml")


def main(_):
  do_not_generate_docs_for = []

  for blocked_doc in do_not_generate_docs_for:
    doc_controls.do_not_generate_docs(blocked_doc)

  doc_generator = generate_lib.DocGenerator(
      root_title="Neural Structured Learning",
      py_modules=[("nsl", nsl)],
      code_url_prefix=_CODE_URL_PREFIX.value,
      search_hints=_SEARCH_HINTS.value,
      site_path=_SITE_PATH.value,
      # local_definitions_filter ensures that shared modules are only
      # documented in the location that defines them, instead of every location
      # that imports them.
      callbacks=[public_api.local_definitions_filter])
  doc_generator.build(output_dir=_OUTPUT_DIR.value)


if __name__ == "__main__":
  app.run(main)
