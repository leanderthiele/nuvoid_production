#!/bin/bash

set -e -o pipefail

# Small script to prepare the simulation pipeline
# Command line arguments:
#   [1] index (for the cosmology)

codebase=$HOME/nuvoid_production

source $codebase/utils.sh

# the index
idx="$1"

# where we write the generated job files
JOB_DIR=$codebase/jobs_cosmo_fiducial
mkdir -p $JOB_DIR

# output file root (i.e. slurm output files)
OUT_ROOT=$JOB_DIR/slurm_$idx

# template files to use
COSMO_TEMPLATE=$codebase/cosmo_fiducial.sh
FASTPM_TEMPLATE=$codebase/jobstep_fastpm.sbatch
ROCKSTAR_TEMPLATE=$codebase/jobstep_rockstar.sbatch
ROCKSTAR_LEFTOVERS_TEMPLATE=$codebase/jobstep_rockstar_leftovers.sbatch
PARENTS_TEMPLATE=$codebase/jobstep_parents.sbatch
SUBMIT_TEMPLATE=$codebase/submit.sh

# write the cosmology file
cosmo_sh=$JOB_DIR/cosmo_fiducial_$idx.sh
cp $COSMO_TEMPLATE $cosmo_sh
utils::replace $cosmo_sh 'idx' "$idx"

# write the fastpm file
fastpm_sbatch=$JOB_DIR/jobstep_fastpm_fiducial_$idx.sbatch
cp $FASTPM_TEMPLATE $fastpm_sbatch
utils::replace $fastpm_sbatch 'cosmo'   "$cosmo_sh"
utils::replace $fastpm_sbatch 'output'  "${OUT_ROOT}_fastpm.out"

# write the rockstar file
rockstar_sbatch=$JOB_DIR/jobstep_rockstar_fiducial_$idx.sbatch
cp $ROCKSTAR_TEMPLATE $rockstar_sbatch
utils::replace $rockstar_sbatch 'cosmo'  "$cosmo_sh"
utils::replace $rockstar_sbatch 'output' "${OUT_ROOT}_rockstar.out"

# write the rockstar leftovers file
rockstar_leftovers_sbatch=$JOB_DIR/jobstep_rockstar_leftovers_fiducial_$idx.sbatch
cp $ROCKSTAR_LEFTOVERS_TEMPLATE $rockstar_leftovers_sbatch
utils::replace $rockstar_leftovers_sbatch 'cosmo'  "$cosmo_sh"
utils::replace $rockstar_leftovers_sbatch 'output' "${OUT_ROOT}_rockstar.out"

# write the parents file
parents_sbatch=$JOB_DIR/jobstep_parents_fiducial_$idx.sbatch
cp $PARENTS_TEMPLATE $parents_sbatch
utils::replace $parents_sbatch 'cosmo'  "$cosmo_sh"
utils::replace $parents_sbatch 'output' "${OUT_ROOT}_parents.out"

# write the submit file
submit_sh=$JOB_DIR/submit_$idx.sh
cp $SUBMIT_TEMPLATE $submit_sh
utils::replace $submit_sh 'jobstep_fastpm'             "$fastpm_sbatch"
utils::replace $submit_sh 'jobstep_rockstar'           "$rockstar_sbatch"
utils::replace $submit_sh 'jobstep_rockstar_leftovers' "$rockstar_leftovers_sbatch"
utils::replace $submit_sh 'jobstep_parents'            "$parents_sbatch"
