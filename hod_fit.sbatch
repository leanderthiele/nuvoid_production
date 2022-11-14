#!/bin/bash

#SBATCH -n 160
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G
#SBATCH -t 05:00:00

#SBATCH --array=1-6

module load anaconda3/2021.11
conda activate galaxies

sim_version='cosmo_varied'
sim_index='19'

srun -n $SLURM_NTASKS python -u hod_fit.py $sim_version $sim_index
