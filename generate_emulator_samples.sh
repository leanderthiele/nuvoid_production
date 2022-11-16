#!/bin/bash

set -e -x -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/utils.sh

# get the available cosmologies
cosmo_avail=($(bash $codebase/emulator_available_cosmos.sh | tr ',' ' '))
Ncosmo=${#cosmo_avail[@]}

function cantor_pairing {
  k1="$1"
  k2="$2"
  a=$(( (k1+k2) * (k1+k2+1) ))
  echo $(( a / 2 + k2 ))
  return 0
}

# unique process id
proc_idx=$(cantor_pairing $SLURM_ARRAY_TASK_ID $SLURM_PROCID)

# loop until time is up
consecutive_fails=0

for ii in $( seq 0 10000 ); do
  run_idx=$(cantor_pairing $proc_idx $ii)
  cosmo_idx=${cosmo_avail[$(( run_idx % Ncosmo ))]}
  # we do not divide by Ncosmo here, otherwise we'll get a grid partially
  hod_idx=$run_idx

  # we do not consider failure a reason to abort
  bash $codebase/generate_emulator_sample.sh $cosmo_idx $hod_idx \
    status=$? || status=$?

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
