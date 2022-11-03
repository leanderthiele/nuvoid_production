#!/bin/bash

set -e -o pipefail

# Wraps the various other hod* codes.
# Command line arguments:
#   [1] working directory
#   [2] the hod hash
#   [3...] the hod description, key=value pairs as requested by hod_galaxies.py
# Prints out the log-likelihood as the last line

# some fixed settings
augments=(0 13 41 89) # arbitrary
vide_out='untrimmed_dencut' # this is what Alice recommends
boss_voids='/tigress/lthiele/boss_dr12/voids/sample_test'

Rmin=30 # could be a bit on the low side
Rmax=80 # with this choice the last few bins are basically empty in the data
Nbins=32

codebase="$HOME/nuvoid_production"

source $codebase/utils.sh

wrk_dir="$1"
hod_hash="$2"
hod_desc="${@:3}"

mkdir -p "$wrk_dir/hod/$hod_hash"

if [ -z $SLURM_CPUS_PER_TASK ]; then
  # head node
  export OMP_NUM_THREADS=4
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# generate galaxies at each redshift
bash $codebase/hod_galaxies.sh $wrk_dir $hod_hash $hod_desc

# generate the lightcones
for augment in ${augments[@]}; do
  bash $codebase/hod_lightcone.sh $wrk_dir $hod_hash $augment 'rockstar'
done

# generate the void catalogs
for augment in ${augments[@]}; do
  vide_log="$wrk_dir/hod/$hod_hash/vide_$augment.log"
  # VIDE fails in rare cases pretty randomly, in such cases retry
  vide_max_tries=3
  vide_failed=1
  for retry in $(seq 1 $vide_max_tries); do
    utils::run "bash $codebase/hod_vide.sh $wrk_dir $hod_hash $augment $((retry-1))" $vide_log \
      && status=$? || status=$?
    if [ $status -eq 0 ]; then
      vide_failed=0
      break
    fi
  done
  if [ $vide_failed -eq 1 ]; then
    utils::printerr "VIDE did not succeed within $vide_max_tries tries for $wrk_dir-$hod_hash-$augment"
    exit 1
  fi
done

# measure the data histogram
boss_counts="$(bash $codebase/hod_histogram.sh $boss_voids $vide_out $Rmin $Rmax $Nbins)"

# measure the simulation histograms and compute their individual log-likelihoods
module load gsl/2.6
loglikes=()
for augment in ${augments[@]}; do
  sim_counts="$(bash $codebase/hod_histogram.sh $wrk_dir/hod/$hod_hash/sample_$augment $vide_out $Rmin $Rmax $Nbins)"
  loglikes+=($($codebase/vsf_like $Nbins $(echo $boss_counts | tr ',' ' ') $(echo $sim_counts | tr ',' ' ')))
done

# combine to compute the total log-likelihood
loglike=$($codebase/vsf_combine_like ${#augments[@]} ${loglikes[@]})
module rm gsl/2.6

# clean up
bash $codebase/hod_cleanup.sh $wrk_dir $hod_hash

# for reference, write the log-likelihoods into the hod directory
loglike_info="$wrk_dir/hod/$hod_hash/loglike.info"
echo "loglike_tot=$loglike" > $loglike_info
for ii in $( seq 0 $(( ${#augments[@]} - 1 )) ); do
  echo "loglike_augment${augments[$ii]}=${loglikes[$ii]}" >> "$loglike_info"
done
