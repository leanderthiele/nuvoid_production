#!/bin/bash

#SBATCH -N 8
#SBATCH --ntasks-per-node=40
#SBATCH -c 1
#SBATCH --mem=185G

#SBATCH -t 03:00:00

#SBATCH --array=1-4

set -e -x -o pipefail

# =========== CONFIGURATION OPTIONS ==========
# these are really the only things we want to change from sbatch to sbatch script

# use this to uniquely identify runs, replace by array or similar later
export ID="test1"

# size of the simulation
export NC=2800

# random FastPM seed
export SEED=137

# =========== GENERAL QUIJOTE =============
# exports BOX_SIZE, Z_OUT, BASE
source cmass.sh

# =========== COSMOLOGY =============
# exports all COSMO_* variables
source mnu200meV_cosmo.sh

# =========== RUN CODES ===========
bash run_rockstar_batch.sh $SLURM_ARRAY_TASK_COUNT $((SLURM_ARRAY_TASK_ID-SLURM_ARRAY_TASK_MIN))
