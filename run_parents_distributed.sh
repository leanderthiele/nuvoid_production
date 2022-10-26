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

# because the later times have systematically more halos which increases runtime
# for the parents code, it is better to do some reordering here
myindices=()
for i in $( seq 0 $(( ${#todo[@]} - 1 )) ); do
  if [ $(( (i/world_size) % 2 )) -eq 1 ]; then
    # odd portion
    if [ $(( i%world_size )) -eq $rank ]; then
      myindices+=(${todo[$i]})
    fi
  else
    # even portion
    if [ $(( world_size - (i%world_size) - 1 )) -eq $rank ]; then
      myindices+=(${todo[$i]})
    fi
  fi
done

for i in ${myindices[@]}; do
  bash $codebase/run_parents.sh $i
done
