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
# all snapshots:
# 0.5965 0.6074 0.6152 0.6216 0.6270 0.6320 0.6367 0.6409 0.6449 0.6488 0.6526 0.6563 0.6601 0.6638 0.6676 0.6715 0.6757 0.6803 0.6856 0.6930
source globals.sh
./lightcone \
  "/scratch/gpfs/lthiele/nuvoid_production/test1" \
  "fidhod" \
  "test" \
  $BOX_SIZE \
  0.25 \
  0.42 0.70 \
  0 \
  /tigress/lthiele/boss_dr12 \
  1 \
  0.6152 0.6488 0.6715
