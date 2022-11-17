#!/bin/bash

codebase="$HOME/nuvoid_production"

function submit_batch {
  mode="$1"
  max_simultaneous="$2"

  start_idx="$3"
  end_idx="$4"
  specific_queue="$5"
  first_dependence="$6"

  if [ -z $specific_queue ]; then
    echo "Submitting for all queues"
    queues=$( seq 1 $max_simultaneous )
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
    for i in $( seq $((start_idx+queue-1)) $max_simultaneous $end_idx ); do
      # use tail here so we don't capture some potential other printouts
      fastpm_dependency="$(bash $codebase/jobs_cosmo_$mode/submit_$i.sh $fastpm_dependency | tail -1)"
    done
    echo "Finished building queue $queue / $max_simultaneous"
  done
}
