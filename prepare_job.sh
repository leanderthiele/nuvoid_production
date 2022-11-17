#!/bin/bash

codebase=$HOME/nuvoid_production
source $codebase/utils.sh

function prepare_job {
  mode="$1"
  idx="$2"

  # check mode is valid
  if ! [[ "$mode" =~ ^('fiducial'|'varied')$ ]]; then return 1; fi

  # where we write the generated job files
  JOB_DIR="$codebase/jobs_cosmo_$mode"
  mkdir -p $JOB_DIR

  # output file root (i.e. slurm output files)
  OUT_ROOT=$JOB_DIR/slurm_$idx

  # the script we use to draw a cosmology
  COSMO_DRAW_EXE="bash $codebase/draw_cosmo.sh"

  # template files to use
  COSMO_TEMPLATE=$codebase/cosmo_$mode.sh
  FASTPM_TEMPLATE=$codebase/jobstep_fastpm.sbatch
  ROCKSTAR_TEMPLATE=$codebase/jobstep_rockstar.sbatch
  ROCKSTAR_LEFTOVERS_TEMPLATE=$codebase/jobstep_rockstar_leftovers.sbatch
  PARENTS_TEMPLATE=$codebase/jobstep_parents.sbatch
  SUBMIT_TEMPLATE=$codebase/submit.sh

  # FIXME write the cosmology file
  cosmo_sh=$JOB_DIR/cosmo_fiducial_$idx.sh
  cp $COSMO_TEMPLATE $cosmo_sh
  utils::replace $cosmo_sh 'idx' "$idx"
  if [ "$mode" = 'fiducial' ]; then
    :
  elif [ "$mode" = 'varied' ]; then
    cosmo="$($COSMO_DRAW_EXE $idx)"
    read Omega_b Omega_m h0 A_s n_s M_nu <<< "$(echo "$cosmo" | tr ',' ' ')"
    utils::replace $cosmo_sh 'Omega_m' "$Omega_m"
    utils::replace $cosmo_sh 'Omega_b' "$Omega_b"
    utils::replace $cosmo_sh 'h0'      "$h0"
    utils::replace $cosmo_sh 'n_s'     "$n_s"
    utils::replace $cosmo_sh 'A_s'     "$A_s"
    utils::replace $cosmo_sh 'M_nu'    "$M_nu"
  else
    return 1
  fi

  # write the fastpm file
  fastpm_sbatch=$JOB_DIR/jobstep_fastpm_${mode}_$idx.sbatch
  cp $FASTPM_TEMPLATE $fastpm_sbatch
  utils::replace $fastpm_sbatch 'cosmo'   "$cosmo_sh"
  utils::replace $fastpm_sbatch 'output'  "${OUT_ROOT}_fastpm.out"

  # write the rockstar file
  rockstar_sbatch=$JOB_DIR/jobstep_rockstar_${mode}_$idx.sbatch
  cp $ROCKSTAR_TEMPLATE $rockstar_sbatch
  utils::replace $rockstar_sbatch 'cosmo'  "$cosmo_sh"
  utils::replace $rockstar_sbatch 'output' "${OUT_ROOT}_rockstar.out"

  # write the rockstar leftovers file
  rockstar_leftovers_sbatch=$JOB_DIR/jobstep_rockstar_leftovers_${mode}_$idx.sbatch
  cp $ROCKSTAR_LEFTOVERS_TEMPLATE $rockstar_leftovers_sbatch
  utils::replace $rockstar_leftovers_sbatch 'cosmo'  "$cosmo_sh"
  utils::replace $rockstar_leftovers_sbatch 'output' "${OUT_ROOT}_rockstar.out"

  # write the parents file
  parents_sbatch=$JOB_DIR/jobstep_parents_${mode}_$idx.sbatch
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

  return 0
}
