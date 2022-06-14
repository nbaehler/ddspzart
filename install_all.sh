#/bin/bash

# sudo apt-get install make gcc g++ llvm-8 ffmpeg libsndfile-dev

# path=$(pwd)
# cd /usr/bin
# sudo ln -s llvm-config-8 llvm-config
# cd $path

# export LLVM_CONFIG_PATH=/usr/bin/llvm-config

cd omnizart
make install
pip install numpy
pip install pyFluidSynth
pip install platformdirs
pip install click==7.1.2

# Download from
# https://drive.google.com/uc?export=download&id=10i8z1zH60a2coKEst47lELdkvZUmgd1b
# Unzip and place content in omnizart/tests/resource/

make install-dev

echo "Omnizart installed successfully (maybe)!"

cd ../ddsp

./update_pip.sh
pip install --upgrade pip
pip install --upgrade ddsp
pip install --upgrade tensorflow

echo "DDSP installed successfully (maybe)!"

# cd ../spleeter

# pip install poetry
# poetry install

# echo "Spleeter installed successfully (maybe)!"

# cd ..

# pip install tf-nightly
# pip install --upgrade tensorflow
# pip install --upgrade keras
