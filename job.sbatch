#!/bin/bash

#SBATCH -N 70
#SBATCH --ntasks-per-node=40
#SBATCH -c 1
#SBATCH --mem=185G

#SBATCH -t 01:00:00

set -e -x -o pipefail

# =========== CONFIGURATION OPTIONS ==========
# these are really the only things we want to change from sbatch to sbatch script

# use this to uniquely identify runs, replace by array or similar later
export ID="test1"

# size of the simulation
export NC=2800

# how many parallel Rockstar processes we allow in maximum (due to memory constraints perhaps)
# If this is set to zero or not defined, the number of nodes is used
export NMAX_PARALLEL_ROCKSTAR=0

# =========== GENERAL QUIJOTE =============
# exports BOX_SIZE, Z_OUT, BASE
source cmass.sh

# =========== COSMOLOGY =============
# exports all COSMO_* variables
source mnu200meV_cosmo.sh

# =========== INPUT FILE TEMPLATES =========
# exports all *_TEMPLATE variables
source fid_templates.sh

# =========== RUN ALL CODES ===========
bash run_reps_fastpm_rockstar.sh
