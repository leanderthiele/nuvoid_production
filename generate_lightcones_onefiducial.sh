#!/bin/bash

set -e -o pipefail

# Command line arguments:
#   [1] cosmo fiducial index
#   [2] fiducial hod version

codebase=$HOME/nuvoid_production

source $codebase/utils.sh

# our small driver to interact with the database
MYSQL_EXE="$codebase/mysql_driver"

seed_idx="$1"
hod_version="$2"

# these are coming from the voids posterior
if [ $hod_version -eq 0 ]; then
  HOD_FID=(
           'hod_transfP1=1.5'
           'hod_abias=-0.5'
           'hod_log_Mmin=12.73'
           'hod_sigma_logM=0.3'
           'hod_log_M0=14.4'
           'hod_log_M1=14.4'
           'hod_alpha=0.6'
           'hod_transf_eta_cen=6.0'
           'hod_transf_eta_sat=-0.5'
           'hod_mu_Mmin=-2.0'
           'hod_mu_M1=-10.0'
          )
else
  exit 1
fi

# construct the HOD description command line
hod_desc='cat=rockstar secondary=kinpot have_vbias=True have_zdep=True'
for ii in $( seq 0 $(( ${#HOD_FID[@]} - 1 )) ); do
  hod_desc="$hod_desc ${HOD_FID[$ii]}"
done

# compute the HOD hash
hod_hash="$(utils::hex_hash "$hod_desc")"

# we start computing at 0, it doesn't really matter
augment_idx=0

data_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_fiducial_${seed_idx}"

wrk_dir="/tmp/cosmo_fiducial_${seed_idx}"
hod_dir="$wrk_dir/lightcones/${hod_hash}"
mkdir -p $hod_dir

export OMP_NUM_THREADS=1 # I think we need to do this to avoid OOM
bash $codebase/lightcones_galaxies.sh $data_dir $wrk_dir $hod_hash $hod_desc

if [ -z $SLURM_CPUS_PER_TASK ]; then
  export OMP_NUM_THREADS=4
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# generate all augmentations
bash $codebase/lightcones_lightcone.sh $data_dir $wrk_dir $hod_hash $augment_idx 'rockstar' 96

# now we have the lightcone file and shouldn't need to use the data_dir anymore
bash $codebase/lightcones_cleanup.sh $wrk_dir $hod_hash

# if successful, copy into permanent storage
target_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_fiducial_${seed_idx}/lightcones"
mkdir -p $target_dir
mv "$hod_dir" "$target_dir"
