#!/bin/bash
#
# Call with command line arguments:
#   [1] first index to be submitted
#   [2] last index to be submitted
#   [3] (optional) only work on the queue starting at index [1]
#                  Any argument can be given.
#   [4] (optional, only if [3] is given)
#                  Dependence for the first fastpm job

set -e -o pipefail

codebase="$HOME/nuvoid_production"
source $codebase/submit_batch.sh

submit_batch 'varied' 1 "$@"
