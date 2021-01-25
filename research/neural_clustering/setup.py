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
"""Setup file for Neural Clustering."""

import setuptools

package_name = "neural_clustering"
packages = [package_name]
packages += [package_name + "." + x for x in setuptools.find_packages()]

setuptools.setup(
    name=package_name,
    version="0.1",
    description="A Tensorflow package of supervised amortized clustering using neural networks",
    packages=packages,
    package_dir={package_name: ""},
)
