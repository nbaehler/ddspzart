#/bin/bash
deactivate
source deactivate
source ../.venv/bin/activate

export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES='-1'

python omni_transcribe.py
python ddsp_timbre_transfer.py
