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

. /load.sh

omnizart drum transcribe sample.wav
omnizart chord transcribe sample.wav
omnizart music transcribe sample.wav

echo FINISHED at $(date)
