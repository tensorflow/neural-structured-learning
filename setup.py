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
"""Setup for TensorFlow Neural Structured Learning package."""

import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

INSTALL_REQUIRES = [
    "absl-py",
    "attrs",
    "scipy",
    "six",
]

setuptools.setup(
    name="neural-structured-learning",
    version="1.0.0",
    author="Google LLC",
    description="TensorFlow Neural Structured Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorflow/neural-structured-learning",
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

# attrs
