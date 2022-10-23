#!/bin/bash

set -e -o pipefail

# the sbatch scripts, to be populated
jobstep_fastpm=<<<jobstep_fastpm>>>
jobstep_rockstar=<<<jobstep_rockstar>>>
jobstep_parents=<<<jobstep_parents>>>

function get_jobid {
  len=7 # expected length of the slurm jobids
  jobid=$(echo "$1" | grep -m 1 -oP "Submitted\sbatch\sjob\s+\K\d{$len}")
  if [ ${#jobid} -ne $len ]; then
    return 1
  fi
  return 0
}

result="$(sbatch $jobstep_fastpm)"
fastpm_jobid=$(get_jobid "$result")

result="$(sbatch --dependency=afterok:$fastpm_jobid $jobstep_rockstar)"
rockstar_jobid=$(get_jobid "$result")

result="$(sbatch --dependency=afterok:$rockstar_jobid $jobstep_parents)"
parents_jobid=$(get_jobid "$result")
