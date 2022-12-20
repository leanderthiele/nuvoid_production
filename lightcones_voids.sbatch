#!/bin/bash

#SBATCH -n 160
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00
#SBATCH --array=1-8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

codebase="$HOME/nuvoid_production"

srun -n $SLURM_NTASKS bash "$codebase/lightcones_voids.sh"
