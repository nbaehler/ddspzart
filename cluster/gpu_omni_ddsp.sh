#!/bin/bash
#SBATCH --chdir /home/nbaehler/workspace/ddspzart/workspace

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:0:0
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:2

# https://www.epfl.ch/research/facilities/scitas/hardware/izar/

echo STARTING AT $(date)

. ../cluster/load.sh

source ../.omni_venv/bin/activate
python omni_transcribe.py
source deactivate

source ../.ddsp_venv/bin/activate
python ddsp_timbre_transfer.py
source deactivate

echo FINISHED at $(date)
