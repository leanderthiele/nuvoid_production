#!/bin/bash

set -e -o pipefail

# Wraps the various other hod* codes.
# Command line arguments:
#   [1] working directory
#   [2] the hod description, key=value pairs as requested by hod_galaxies.py
# Prints out the log-likelihood as the last line,
# but be careful, there could be other printouts (warnings...) before so
# that needs to be dealt with!

# some fixed settings
augments=(0 13 41 89) # arbitrary
vide_out='untrimmed_dencut' # TODO!!!
boss_voids='/tigress/lthiele/boss_dr12/voids/sample_test'

# TODO
Rmin=20
Rmax=80
Nbins=32

codebase="$HOME/nuvoid_production"

source $codebase/utils.sh

wrk_dir="$1"
hod_desc="${@:2}"

hod_hash="$(utils::hex_hash "$hod_desc")"
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
  utils::run "bash $codebase/hod_vide.sh $wrk_dir $hod_hash $augment" $vide_log
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
module rm gsl/2.6

# combine to compute the total log-likelihood
loglike=$($codebase/vsf_combine_like ${#augments[@]} ${loglikes[@]})

# clean up
bash $codebase/hod_cleanup.sh $wrk_dir $hod_hash

# output
echo $loglike