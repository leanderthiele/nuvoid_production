#!/bin/bash

set -e -o pipefail

codebase="$HOME/nuvoid_production"

MYSQL_DRIVER="$codebase/mysql_driver"

# these take input, output directory as command line arguments
BIN_TXT="python $codebase/lightcones_convert_bin_txt.py"
VIDE_SH="bash $codebase/lightcones_vide.sh"

# loop until time is up
for ii in $( seq 1 10000 ); do
  # get work
  read cosmo_idx hod_idx hod_hash <<< "$($MYSQL_DRIVER 'create_voids')"
  if [ $cosmo_idx -lt 0 ]; then
    break
  fi

  # we are starting
  $MYSQL_DRIVER 'start_voids' $cosmo_idx $hod_idx

  perm_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_${cosmo_idx}/lightcones/${hod_hash}"
  tmp_dir="/tmp/cosmo_varied_${cosmo_idx}/lightcones/${hod_hash}"

  # convert .bin -> .txt
  module load anaconda3/2021.11
  $BIN_TXT $perm_dir $tmp_dir \
    && status=$? || status=$?
  module rm anaconda3/2021.11
  if [ $status -ne 0 ]; then
    $MYSQL_DRIVER 'end_voids' $cosmo_idx $hod_idx $status
    continue
  fi

  # find voids
  $VIDE_SH $tmp_dir $perm_dir \
    && status=$? || status=$?
  $MYSQL_DRIVER 'end_voids' $cosmo_idx $hod_idx $status
done
