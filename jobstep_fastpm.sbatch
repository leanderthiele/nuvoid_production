#!/bin/bash

#SBATCH -N 70
#SBATCH --ntasks-per-node=40
#SBATCH -c 1
#SBATCH --mem=185G

#SBATCH -t 03:00:00

#SBATCH -o <<<output>>>

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/cmass.sh
source <<<cosmo>>>

bash $codebase/run_class_reps.sh
bash $codebase/run_fastpm.sh
