#!/bin/bash

#SBATCH -n 160
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00

#SBATCH --array=1-16

module load anaconda3/2021.11
conda activate nbodykit

export OMP_NUM_THREADS=1
export CPUS_PER_TASK=3

codebase="/home/lthiele/nuvoid_production"
nodelist="$("$codebase/expand_nodelist" "$SLURM_JOB_NODELIST")"

mpirun -n $SLURM_NTASKS -hosts="$nodelist" python fiducials_lightcones_plk.py 0
