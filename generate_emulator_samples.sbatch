#!/bin/bash

#SBATCH -n 160
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00

#SBATCH --array=18-30

#SBATCH --signal=ALRM@30

# NOTE it is very important that the array indices are kept unique!!!

srun -n $SLURM_NTASKS bash generate_emulator_samples.sh
