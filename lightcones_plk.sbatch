#!/bin/bash

#SBATCH -n 40
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 01:00:00

module load anaconda3/2021.11
conda activate nbodykit

export OMP_NUM_THREADS=1
export CPUS_PER_TASK=2

# TODO need to use the -hosts argument here becuase the
#      MPI shipped with this conda is buggy...
mpirun -n $SLURM_NTASKS python lightcones_plk.py
