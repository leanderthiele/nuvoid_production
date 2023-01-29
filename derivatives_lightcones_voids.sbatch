#!/bin/bash

#SBATCH -n 80
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00
#SBATCH --array=13-33

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

codebase="$HOME/nuvoid_production"

version=0

srun -n $SLURM_NTASKS --wait=0 --kill-on-bad-exit=0 bash "$codebase/individual_lightcones_voids.sh" $version 'derivatives'
