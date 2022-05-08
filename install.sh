#/bin/bash

sudo apt-get install make gcc g++ llvm-8 ffmpeg libsndfile-dev

path=$(pwd)
cd /usr/bin
sudo ln -s llvm-config-8 llvm-config
cd $path

export LLVM_CONFIG_PATH=/usr/bin/llvm-config

cd omnizart
make install
source ../.venv/bin/activate
pip install numpy
pip install platformdirs

# Download from
# https://drive.google.com/uc?export=download&id=10i8z1zH60a2coKEst47lELdkvZUmgd1b
# Unzip and place content in omnizart/tests/resource/

make install-dev

cd ../ddsp

./update_pip.sh
pip install --upgrade pip
pip install --upgrade ddsp

cd ../spleeter

pip install poetry
poetry install

cd ..
