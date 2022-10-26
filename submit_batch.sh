#!/bin/bash
#
# Call with command line arguments:
#   [1] first index to be submitted
#   [2] last index to be submitted
#   [3] (optional) only work on the queue starting at index [1]
#                  Any argument can be given.

set -e -o pipefail

# how many fastpm jobs we allow to run simultaneously
MAX_SIMULTANEOUS=2

start_idx=$1
end_idx=$2
specific_queue=$3

if [ -z $specific_queue ]; then
  echo "Submitting for all queues"
  queues=$( seq 1 $MAX_SIMULTANEOUS )
else
  echo "Only submitting one queue"
  queues=1
fi
  

for queue in $queues; do
  fastpm_dependency=""
  for i in $( seq $((start_idx+queue-1)) $MAX_SIMULTANEOUS $end_idx ); do
    fastpm_dependency=$(bash jobs_cosmo_varied/submit_$i.sh $fastpm_dependency)
  done
  echo "Finished building queue $queue / $MAX_SIMULTANEOUS"
done
