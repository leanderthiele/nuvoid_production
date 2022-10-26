#!/bin/bash
#
# Call with command line arguments:
#   [1] first index to be submitted
#   [2] last index to be submitted

set -e -o pipefail

# how many fastpm jobs we allow to run simultaneously
MAX_SIMULTANEOUS=2

start_idx=$1
end_idx=$2

for queue in $( seq 1 $MAX_SIMULTANEOUS ); do
  fastpm_dependency=""
  for i in $( seq $((start_idx+queue-1)) $MAX_SIMULTANEOUS $end_idx ); do
    fastpm_dependency=$(bash jobs_cosmo_varied/submit_$i.sh $fastpm_dependency)
  done
  echo "Finished building queue $queue / $MAX_SIMULTANEOUS"
done
