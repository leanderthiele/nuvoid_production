#!/bin/bash

# Command line arguments:
#    [1] fiducial hod version (integer)

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/utils.sh

hod_version="$1"

consecutive_fails=0

# loop until time is up
for ii in $( seq 0 10000 ); do
  
  # we do not consider failure a reason to abort
  bash $codebase/generate_lightcones_onefiducial.sh $hod_version \
    && status=$? || status=$?

  if [ $status -ne 0 ]; then
    consecutive_fails=$((consecutive_fails+1))
    if [ $consecutive_fails -gt 2 ]; then
      # something must be going terribly wrong
      exit 1
    fi
  else
    consecutive_fails=0
  fi
done
