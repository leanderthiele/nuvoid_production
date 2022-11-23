#!/bin/bash

#SBATCH -n 160
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G
#SBATCH -t 24:00:00

#SBATCH --array=100-120

# NOTE it is very important that the array indices are kept unique!!!

module load openmpi/gcc/3.1.5/64

mpirun -n $SLURM_NTASKS bash generate_emulator_samples.sh
