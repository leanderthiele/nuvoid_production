#!/bin/bash

#SBATCH -n 20
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -t 02:00:00

module load anaconda3/2021.11
conda activate galaxies

sim_version='cosmo_varied'
sim_index='30'

srun -n $SLURM_NTASKS python -u hod_fit.py $sim_version $sim_index
