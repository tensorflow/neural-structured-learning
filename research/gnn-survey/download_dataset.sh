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
# TODO(ppham27): Consider consolidating with
# examples/preprocess/cora/prep_data.sh.
# URL for downloading Cora dataset.
URL=https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

# Target folder to store and process data.
DATA_DIR=data

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
