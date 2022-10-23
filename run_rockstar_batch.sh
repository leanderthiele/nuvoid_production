#!/bin/bash

# Runs Rockstar on multiple snapshots.
# Command line arguments:
# [1] number of jobs in my team
# [2] my index within the team

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/utils.sh
source $codebase/globals.sh

world_size="$1"
rank="$2"

for snap_idx in $( seq $rank $world_size $((NUM_SNAPS-1)) ); do
  bash $codebase/run_rockstar.sh $snap_idx
done
