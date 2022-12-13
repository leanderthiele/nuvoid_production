#!/bin/bash

#SBATCH -n 160
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00

#SBATCH --array=9-20

srun -n $SLURM_NTASKS bash generate_lightcones_samples.sh
