#!/bin/bash

#SBATCH -n 80
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 01:00:00
#SBATCH --array=1-1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

codebase="$HOME/nuvoid_production"

version=0

srun -n $SLURM_NTASKS bash "$codebase/fiducials_lightcones_voids.sh" $version
