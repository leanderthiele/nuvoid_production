#!/bin/bash

set -e -o pipefail

# Command line arguments:
#   [1] cosmo varied index
#   [2] index in the HOD QRNG sequence

codebase=$HOME/nuvoid_production

source $codebase/utils.sh

# our small driver to interact with the database
MYSQL_EXE="$codebase/mysql_driver"


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

cosmo_idx="$1"
hod_idx="$2"

module load gsl/2.6
hod_values=($($codebase/sample_prior $hod_idx 0 "" ${#HOD_KEYS[@]} $codebase/hod_prior.dat | tr ',' ' '))
module rm gsl/2.6

# construct the HOD description command line
hod_desc='cat=rockstar secondary=kinpot have_vbias=True have_zdep=True'
for ii in $( seq 0 $(( ${#HOD_KEYS[@]} - 1 )) ); do
  hod_desc="$hod_desc ${HOD_KEYS[$ii]}=${hod_values[$ii]}"
done

# compute the HOD hash
hod_hash="$(utils::hex_hash "$hod_desc")"

$MYSQL_EXE 'start_trial' $cosmo_idx $hod_idx $hod_hash
echo "Trial [$cosmo_idx, $hod_idx] working on $hod_hash"

# compute the augmentatation index (which is pretty much random)
dec_hash="$(utils::dec_hash "${cosmo_idx}${hod_desc}" 32)"
augment_idx=$((dec_hash % 96))

tmp_dir="/tmp/cosmo_varied_${cosmo_idx}"
data_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_${cosmo_idx}"

wrk_dir="/tmp/cosmo_varied_${cosmo_idx}"
hod_dir="$wrk_dir/lightcones/${hod_hash}"
mkdir -p $hod_dir

export OMP_NUM_THREADS=1 # I think we need to do this to avoid OOM
bash $codebase/lightcones_galaxies.sh $data_dir $wrk_dir $hod_hash $hod_desc

if [ -z $SLURM_CPUS_PER_TASK ]; then
  export OMP_NUM_THREADS=4
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# generates 8 lightcones
bash $codebase/lightcones_lightcone.sh $data_dir $wrk_dir $hod_hash $augment_idx 'rockstar' 8

# now we have the lightcone file and shouldn't need to use the data_dir anymore

bash $codebase/lightcones_cleanup.sh $wrk_dir $hod_hash

# if successful, copy into permanent storage
target_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_${cosmo_idx}/lightcones"
mkdir -p $target_dir
mv "$hod_dir" "$target_dir"
