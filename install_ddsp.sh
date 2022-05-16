#/bin/bash

cd ddsp

python3 -m venv ../.ddsp_venv
source ../.ddsp_venv/bin/activate

./update_pip.sh
pip install --upgrade pip
pip install --upgrade ddsp

echo "DDSP installed successfully (maybe)!"

cd ..

# pip install tf-nightly
# pip install --upgrade tensorflow
# pip install --upgrade keras
