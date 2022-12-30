#!/bin/bash

#SBATCH -n 160
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00
#SBATCH --array=1-4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

codebase="$HOME/nuvoid_production"

srun -n $SLURM_NTASKS --wait=0 --kill-on-bad-exit=0 bash "$codebase/lightcones_voids.sh"
