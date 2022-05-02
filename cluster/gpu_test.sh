#!/bin/bash
#SBATCH --chdir /home/nbaehler/workspace/ddspzart/workspace

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:5:0
#SBATCH --cpus-per-task=4
#SBATCH --partition=debug
#SBATCH --gres=gpu:1

# https://www.epfl.ch/research/facilities/scitas/hardware/fidis/
# https://www.epfl.ch/research/facilities/scitas/hardware/izar/

echo STARTING AT $(date)

. load.sh

omnizart drum transcribe sample.wav
omnizart chord transcribe sample.wav
omnizart music transcribe sample.wav

echo FINISHED at $(date)
