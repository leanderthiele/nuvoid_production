#!/bin/bash

#SBATCH -n 160
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G
#SBATCH -t 24:00:00

#SBATCH --array=25-40

srun -n $SLURM_NTASKS --wait=0 --kill-on-bad-exit=0 bash generate_lightcones_samples.sh
