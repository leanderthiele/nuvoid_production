#!/bin/bash

set -e -o pipefail

# Command line arguments:
#   [1] cosmo varied index
#   [2] index in the HOD QRNG sequence

codebase=$HOME/nuvoid_production

source $codebase/utils.sh

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

echo "Trial [$cosmo_idx, $hod_idx] working on $hod_hash"

# compute the augmentatation index (which is pretty much random)
dec_hash="$(utils::dec_hash "${cosmo_idx}${hod_desc}" 32)"
augment_idx=$((dec_hash % 96))

# we start working while copying is still happening
tmp_dir="/tmp/cosmo_varied_${cosmo_idx}"
copy_finish_marker="${tmp_dir}/FINISHED_COPY"

if [ -f "$copy_finish_marker" ]; then
  data_dir="/tmp/cosmo_varied_${cosmo_idx}"
else
  data_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_${cosmo_idx}"
fi

wrk_dir="/tmp/cosmo_varied_${cosmo_idx}"
hod_dir="$wrk_dir/emulator/${hod_hash}"
mkdir -p $hod_dir

export OMP_NUM_THREADS=1 # I think we need to do this to avoid OOM
bash $codebase/emulator_galaxies.sh $data_dir $wrk_dir $hod_hash $hod_desc

if [ -z $SLURM_CPUS_PER_TASK ]; then
  export OMP_NUM_THREADS=4
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

bash $codebase/emulator_lightcone.sh $data_dir $wrk_dir $hod_hash $augment_idx 'rockstar'

# now we have the lightcone file and shouldn't need to use the data_dir anymore

bash $codebase/emulator_cleanup.sh $wrk_dir $hod_hash 0

vide_log="$hod_dir/vide_${augment_idx}.log"
utils::run "bash $codebase/emulator_vide.sh $wrk_dir $hod_hash $augment_idx 0" $vide_log \
  && status=$? || status=$?

if [ $status -ne 0 ]; then
  # occasional VIDE failure, we do not care, but should clean up afterwards
  rm -r "$hod_dir"
  exit 1
fi

# VIDE was successful, clean spurious stuff
bash $codebase/emulator_cleanup.sh $wrk_dir $hod_hash 1

# if successful, copy into permanent storage
target_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_${cosmo_idx}/emulator"
mkdir -p $target_dir
mv "$hod_dir" "$target_dir"
