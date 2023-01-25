#!/bin/bash

# Command line arguments:
#   [1] hod_version

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/utils.sh

# our small driver to interact with the database
MYSQL_EXE="$codebase/mysql_driver"

consecutive_fails=0

hod_version="$1"

# loop until time is up
for ii in $( seq 0 10000 ); do
  
  read cosmo_idx hod_idx <<< "$($MYSQL_EXE 'create_deriv' "$hod_version" "${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID}${SLURM_PROCID}$ii")"

  # we do not consider failure a reason to abort
  bash $codebase/generate_lightcones_onederiv.sh $hod_version $cosmo_idx $hod_idx \
    && status=$? || status=$?

  $MYSQL_EXE 'end_deriv' $hod_version $cosmo_idx $hod_idx $status

  if [ $status -ne 0 ]; then
    consecutive_fails=$((consecutive_fails+1))
    utils::printerr "Trial [$cosmo_idx, $hod_idx] failed"
    if [ $consecutive_fails -gt 2 ]; then
      # something must be going terribly wrong
      exit 1
    fi
  else
    consecutive_fails=0
  fi
done
