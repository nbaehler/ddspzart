#/bin/bash
cd spleeter

python3 -m venv ../.spleeter_venv
source ../.spleeter_venv/bin/activate

pip install poetry
poetry install

echo "Spleeter installed successfully (maybe)!"

cd ..
