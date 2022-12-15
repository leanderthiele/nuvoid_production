#!/bin/bash

#SBATCH -n 80
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 01:00:00

module load anaconda3/2021.11
conda activate nbodykit

export OMP_NUM_THREADS=1
export CPUS_PER_TASK=3

codebase="/home/lthiele/nuvoid_production"
nodelist="$(python "$codebase/expand_nodelist.py" "$SLURM_JOB_NODELIST")"

mpirun -n $SLURM_NTASKS -hosts="$nodelist" python lightcones_plk.py
