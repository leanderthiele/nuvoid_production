#!/bin/bash

#SBATCH -n 4
#SBATCH -c 1
#SBATCH --mem-per-cpu=40G
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lthiele@princeton.edu
#SBATCH --spread-job

#SBATCH -t 02:00:00

#SBATCH -o <<<output>>>

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/cmass.sh
source <<<cosmo>>>

srun -n $SLURM_NTASKS -W 0 bash $codebase/run_parents_distributed.sh
