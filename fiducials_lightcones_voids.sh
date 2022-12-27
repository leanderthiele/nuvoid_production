#!/bin/bash

# Command line arguments:
#   [1] version index

set -e -o pipefail

version="$1"

codebase="$HOME/nuvoid_production"

MYSQL_DRIVER="$codebase/mysql_driver"

# these take input, output directory as command line arguments
BIN_TXT="python $codebase/fiducials_lightcones_convert_bin_txt.py"
VIDE_SH="bash $codebase/fiducials_lightcones_vide.sh"

# loop until time is up
for ii in $( seq 1 10000 ); do
  # get work
  read running_idx seed_idx lightcone_idx hod_hash <<< "$($MYSQL_DRIVER 'create_fiducials_voids' $version)"
  if [ $seed_idx -lt 0 ]; then
    break
  fi

  # we are starting
  $MYSQL_DRIVER 'start_fiducials_voids' $version $running_idx $seed_idx $lightcone_idx

  perm_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_fiducial_${seed_idx}/lightcones/${hod_hash}"
  tmp_dir="/tmp/cosmo_fiducial_${seed_idx}/lightcones/${hod_hash}"
  mkdir -p $tmp_dir

  # convert .bin -> .txt
  module load anaconda3/2021.11
  $BIN_TXT $perm_dir $tmp_dir $lightcone_idx \
    && status=$? || status=$?
  module rm anaconda3/2021.11
  if [ $status -ne 0 ]; then
    $MYSQL_DRIVER 'end_fiducials_voids' $version $running_idx $seed_idx $lightcone_idx $status
    continue
  fi

  # find voids
  $VIDE_SH $tmp_dir $perm_dir $lightcone_idx \
    && status=$? || status=$?
  $MYSQL_DRIVER 'end_fiducials_voids' $version $running_idx $seed_idx $lightcone_idx $status
done
