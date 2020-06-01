#!/bin/bash
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

# URL for downloading Cora dataset.
URL=https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

# Target folder to store and process data.
DATA_DIR=/tmp

# Helper function to download the data.
function download () {
  fileurl=${1}
  filedir=${2}
  filename=${fileurl##*/}
  if [ ! -f ${filename} ]; then
    echo ">>> Downloading '${filename}' from '${fileurl}' to '${filedir}'"
    wget --quiet --no-check-certificate -P ${filedir} ${fileurl}
  else
    echo "*** File '${filename}' exists; no need to download it."
  fi
}

# Download and unzip the dataset. Data will be at '${DATA_DIR}/cora/' folder.
download ${URL} ${DATA_DIR}
tar -C ${DATA_DIR} -xvzf ${DATA_DIR}/cora.tgz

# Pre-process cora dataset. The file 'preprocess_cora_dataset.py' is assumed to
# be located in the current directory.
python $(dirname "$0")/preprocess_cora_dataset.py \
--input_cora_content=${DATA_DIR}/cora/cora.content \
--input_cora_graph=${DATA_DIR}/cora/cora.cites \
--max_nbrs=5 \
--output_train_data=${DATA_DIR}/cora/train_merged_examples.tfr \
--output_test_data=${DATA_DIR}/cora/test_examples.tfr
