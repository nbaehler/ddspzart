#!/bin/bash
#SBATCH --chdir /home/nbaehler/workspace/ddspzart/workspace

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:5:0
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug

# https://www.epfl.ch/research/facilities/scitas/hardware/fidis/

echo STARTING AT $(date)

. ../cluster/cpu_load.sh

source ../.omni_venv/bin/activate
python omni_transcribe.py
source deactivate

echo FINISHED at $(date)
