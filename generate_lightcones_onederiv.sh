#!/bin/bash

set -e -o pipefail

# Command line arguments:
#   [1] fiducial hod version (around which we draw)
#   [2] cosmo varied index
#   [3] index in the HOD QRNG sequence

codebase=$HOME/nuvoid_production

source $codebase/utils.sh

# our small driver to interact with the database
MYSQL_EXE="$codebase/mysql_driver"

hod_version="$1"
cosmo_idx="$2"
hod_idx="$3"

HOD_KEYS=(
          'hod_transfP1'
          'hod_abias'
          'hod_log_Mmin'
          'hod_sigma_logM'
          'hod_log_M0'
          'hod_log_M1'
          'hod_alpha'
          'hod_transf_eta_cen'
          'hod_transf_eta_sat'
          'hod_mu_Mmin'
          'hod_mu_M1'
         )

# draw the HOD
module load gsl/2.6
hod_values=($($codebase/sample_prior $hod_idx 0 "" ${#HOD_KEYS[@]} \
              $codebase/hod_deriv_prior_v${hod_version}.dat | tr ',' ' '))

# construct the HOD description command line
hod_desc='cat=rockstar secondary=kinpot have_vbias=True have_zdep=True'
for ii in $( seq 0 $(( ${#HOD_KEYS[@]} - 1 )) ); do
  hod_desc="$hod_desc ${HOD_KEYS[$ii]}=${hod_values[$ii]}"
done

# compute the HOD hash
hod_hash="$(utils::hex_hash "$hod_desc")"

# indicate we are starting
$MYSQL_EXE 'start_deriv' $hod_version $cosmo_idx $hod_idx $hod_hash

# we start computing at 0, it doesn't really matter
augment_idx=0

data_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_${cosmo_idx}"

wrk_dir="/tmp/cosmo_varied_${cosmo_idx}"
hod_dir="$wrk_dir/lightcones/${hod_hash}"
mkdir -p $hod_dir

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2)) # I think we need to do this to avoid OOM
bash $codebase/lightcones_galaxies.sh $data_dir $wrk_dir $hod_hash $hod_desc \
  && status=$? || status=$?

if [ $status -ne 0 ]; then
  exit 1
fi

if [ -z $SLURM_CPUS_PER_TASK ]; then
  export OMP_NUM_THREADS=4
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# generate all augmentations
bash $codebase/lightcones_lightcone.sh $data_dir $wrk_dir $hod_hash $augment_idx 'rockstar' 96 \
  && status=$? || status=$?

if [ $status -ne 0 ]; then
  exit 1
fi

# now we have the lightcone file and shouldn't need to use the data_dir anymore
bash $codebase/lightcones_cleanup.sh $wrk_dir $hod_hash

# if successful, copy into permanent storage
target_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_${cosmo_idx}/derivatives_v${hod_version}"
mkdir -p $target_dir
mv "$hod_dir" "$target_dir"
