#!/bin/bash

#SBATCH -N 8
#SBATCH --ntasks-per-node=40
#SBATCH -c 1
#SBATCH --mem=185G
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lthiele@princeton.edu

#SBATCH -t 02:30:00

#SBATCH --array=1-4

#SBATCH -o <<<output>>>_%a

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/cmass.sh
source <<<cosmo>>>

bash $codebase/run_rockstar_batch.sh \
  $SLURM_ARRAY_TASK_COUNT $((SLURM_ARRAY_TASK_ID-SLURM_ARRAY_TASK_MIN))
