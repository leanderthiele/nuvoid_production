#!/bin/bash

# Little wrapper to cd into working directory because otherwise
# VIDE being stupid sees the same files from multiple processes.

rank=$SLURM_PROCID

wrkdir=/scratch/gpfs/lthiele/nuvoid_production/test1/voids/logs$rank

mkdir -p $wrkdir
cd $wrkdir

module load anaconda3/2021.11
conda activate galaxies

python -u -m void_pipeline $HOME/nuvoid_production/vide_cfg.py
