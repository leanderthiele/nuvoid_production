#!/bin/bash

#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 05:00:00

#SBATCH --array=1-32

#SBATCH --output=/dev/null --error=/dev/null

module load anaconda3/2021.11
conda activate torch-env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

codebase='/home/lthiele/nuvoid_production'
conf_file="$codebase/boss_full_shape_ours.conf"
params_file='<<<params_file>>>'
wrkdir_trial='<<<wrkdir_trial>>>'
wrkdir_production='<<<wrkdir_production>>>'

sleep $SLURM_ARRAY_TASK_ID

python -u /home/lthiele/montepython_public/montepython/MontePython.py run \
  --conf="$conf_file" \
  --param="$params_file" \
  --output="$wrkdir_production" \
  --covmat="$wrkdir_trial/$(basename $wrkdir_trial).covmat" \
  --bestfit="$wrkdir_trial/$(basename $wrkdir_trial).bestfit" \
  -N 100000
