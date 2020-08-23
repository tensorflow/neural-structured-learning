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
