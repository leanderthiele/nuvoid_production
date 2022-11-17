#!/bin/bash

set -e -o pipefail

# Small script to prepare the simulation pipeline
# Command line arguments:
#   [1] index (for the cosmology)

codebase=$HOME/nuvoid_production

source $codebase/prepare_job.sh

# the index
idx="$1"

prepare_job 'varied' "$idx"
