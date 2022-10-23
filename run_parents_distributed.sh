#!/bin/bash

# Simple wrapper around run_parents.sh to be used with SRUN

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/utils.sh
source $codebase/globals.sh

rank=$SLURM_PROCID
world_size=$SLURM_NTASKS

todo=()
for snap_idx in $( seq 0 $((NUM_SNAPS-1)) ); do
  if [ ! -d "$ROOT/rockstar_${TIMES_ARR[$snap_idx]}/out_${snap_idx}_hosts.bf" ]; then
    todo+=($snap_idx)
  fi
done

for i in $( seq $rank $world_size $(( ${#todo[@]} - 1 )) ); do
  bash $codebase/run_parents.sh ${todo[$i]}
done
