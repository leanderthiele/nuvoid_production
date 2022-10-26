#!/bin/bash
#
# Call with optional command line argument:
#   [1] jobid that fastpm is dependent on

set -e -o pipefail

fastpm_dependency=$1

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
  echo $jobid
  return 0
}

if [ -z $fastpm_dependency ]; then
  dependency_str=""
else
  dependency_str="--dependency=afterok:$fastpm_dependency"
fi
result="$(sbatch $dependency_str $jobstep_fastpm)"
fastpm_jobid=$(get_jobid "$result")

result="$(sbatch --dependency=afterok:$fastpm_jobid $jobstep_rockstar)"
rockstar_jobid=$(get_jobid "$result")

result="$(sbatch --dependency=afterok:$rockstar_jobid $jobstep_parents)"
parents_jobid=$(get_jobid "$result")

# this is the indicator that the next fastpm job can start
# since disk space is free again
echo $rockstar_jobid
