#!/bin/bash

cd ddsp

python3 -m venv ../.ddsp_venv
source ../.ddsp_venv/bin/activate

./update_pip.sh
pip install --upgrade pip
pip install --upgrade ddsp
pip install --upgrade tensorflow

echo "DDSP installed successfully (maybe)!"

cd ..

# pip install gsutil
# curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-386.0.0-linux-x86_64.tar.gz
# tar -xf google-cloud-cli-386.0.0-linux-x86_64.tar.gz
# ./google-cloud-sdk/install.sh
# ./google-cloud-sdk/bin/gcloud init
