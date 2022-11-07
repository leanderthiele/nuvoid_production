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

# how many fastpm jobs we allow to run simultaneously
MAX_SIMULTANEOUS=2

start_idx=$1
end_idx=$2
specific_queue=$3
first_dependence=$4

if [ -z $specific_queue ]; then
  echo "Submitting for all queues"
  queues=$( seq 1 $MAX_SIMULTANEOUS )
else
  echo "Only submitting one queue"
  queues=1
fi
  

for queue in $queues; do
  if [ -z $first_dependence ]; then
    fastpm_dependency=""
  else
    fastpm_dependency=$first_dependence
  fi
  for i in $( seq $((start_idx+queue-1)) $MAX_SIMULTANEOUS $end_idx ); do
    # use tail here so we don't capture some potential other printouts
    fastpm_dependency=$(bash jobs_cosmo_fiducial/submit_$i.sh $fastpm_dependency | tail -1)
  done
  echo "Finished building queue $queue / $MAX_SIMULTANEOUS"
done
