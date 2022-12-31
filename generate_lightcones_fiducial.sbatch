#!/bin/bash

#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00

#SBATCH --array=8-8

hod_version=0
srun -n $SLURM_NTASKS --wait=0 --kill-on-bad-exit=0 bash generate_lightcones_fiducial.sh $hod_version
