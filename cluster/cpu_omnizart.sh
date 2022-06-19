#!/bin/bash
#SBATCH --chdir /home/nbaehler/workspace/ddspzart/workspace

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:0:0
#SBATCH --cpus-per-task=28

# https://www.epfl.ch/research/facilities/scitas/hardware/fidis/

echo STARTING AT $(date)

. ../cluster/cpu_load.sh

source ../.venv/bin/activate
python omni_transcribe.py
source deactivate

echo FINISHED at $(date)
