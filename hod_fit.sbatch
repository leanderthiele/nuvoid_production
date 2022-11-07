#!/bin/bash

#SBATCH -n 40
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00

#SBATCH --array=7-9

module load anaconda3/2021.11
conda activate galaxies

sim_version='cosmo_varied'
sim_index='19'

srun -n $SLURM_NTASKS python -u hod_fit.py $sim_version $sim_index
