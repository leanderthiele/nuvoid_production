#!/bin/bash

#SBATCH -n 20
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00

#SBATCH --array=1-16

hod_version=0

srun -n $SLURM_NTASKS --wait=0 --kill-on-bad-exit=0 bash generate_lightcones_derivs.sh $hod_version
