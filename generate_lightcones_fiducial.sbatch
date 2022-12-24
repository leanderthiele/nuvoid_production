#!/bin/bash

#SBATCH -n 10
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00

#SBATCH --array=1-7

hod_version=0
srun -n $SLURM_NTASKS --wait=0 --kill-on-bad-exit=0 bash generate_lightcones_fiducial.sh $hod_version
