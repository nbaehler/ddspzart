#!/bin/bash
#SBATCH --chdir /home/nbaehler/workspace/ddspzart/workspace

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:0:0
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1

# https://www.epfl.ch/research/facilities/scitas/hardware/izar/

echo STARTING AT $(date)

. ../cluster/gpu_load.sh

source ../.venv/bin/activate
python omni_transcribe.py
source deactivate

echo FINISHED at $(date)
