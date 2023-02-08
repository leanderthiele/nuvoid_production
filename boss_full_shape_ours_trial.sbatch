#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00

module load anaconda3/2021.11
module rm anaconda3/2021.11
module load anaconda3/2021.11
conda activate class-pt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

codebase='/home/lthiele/nuvoid_production'
params_file="<<<params_file>>>"
wrkdir="<<<wrkdir>>>"
conf_file="$codebase/boss_full_shape_ours.conf"

python -u /home/lthiele/montepython_public/montepython/MontePython.py run \
  --conf="$conf_file" \
  --param="$params_file" \
  --output="$wrkdir" \
  --jumping=sequential \
  -N 100000
