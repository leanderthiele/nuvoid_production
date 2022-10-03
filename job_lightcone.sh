#!/bin/bash

#SBATCH -N 8
#SBATCH --ntasks-per-node=40
#SBATCH -c 1
#SBATCH --mem=185G

#SBATCH -t 01:00:00

#SBATCH --array=1-2

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
source globals.sh
./lightcone \
  "/scratch/gpfs/lthiele/nuvoid_production/test1" \
  "fidhod" \
  "allsnaps" \
  $BOX_SIZE \
  $COSMO_OMEGA_M \
  0.42 0.70 \
  0 \
  $TIMES
