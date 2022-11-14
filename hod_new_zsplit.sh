#!/bin/bash

set -e -o pipefail

# Command line arguments:
#   [1] hod hash working directory
# Prints out the new total log-likelihood

vide_out='untrimmed_dencut' # this is what Alice recommends
boss_voids='/tigress/lthiele/boss_dr12/voids/sample_test'

Nbins=32
zedges=0.53

codebase="$HOME/nuvoid_production"

wrk_dir="$1"

# overall number of histogram bins, taking into account possible redshift splits
if [ -z $zedges ]; then
  total_bins=$Nbins
else
  commas="${zedges//[^,]}"
  Ncommas="${#commas}"
  Nzbins=$(( Ncommas + 2 ))
  total_bins=$(( Nbins * Nzbins ))
fi

# measure the data histogram
boss_counts="$(bash $codebase/hod_histogram.sh $boss_voids $vide_out $Rmin $Rmax $Nbins $zedges)"

module load gsl/2.6
loglikes=()
for d in $wrk_dir/sample_*; do
  sim_counts="$(bash $codebase/hod_histogram.sh $d $vide_out $Rmin $Rmax $Nbins $zedges)"
  loglikes+=($($codebase/vsf_like $total_bins $(echo $boss_counts | tr ',' ' ') $(echo $sim_counts | tr ',' ' ')))
done

loglike=$($codebase/vsf_combine_like ${#loglikes[@]} ${loglikes[@]})

module rm gsl/2.6

echo $loglike
