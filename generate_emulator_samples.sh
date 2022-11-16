#!/bin/bash

set -e -o pipefail

codebase=$HOME/nuvoid_production

# get the available cosmologies
cosmo_avail=($(bash $codebase/emulator_available_cosmos.sh | head -1 | tr ',' ' '))
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
for ii in $( seq 0 10000 ); do
  run_idx=$(cantor_pairing $proc_idx $ii)
  cosmo_idx=${cosmo_avail[$(( run_idx % Ncosmo ))]}
  hod_idx=$(( run_idx / Ncosmo ))
  bash $codebase/generate_emulator_sample.sh $cosmo_idx $hod_idx
done
