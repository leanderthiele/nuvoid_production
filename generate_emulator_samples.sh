#!/bin/bash

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/utils.sh

# figure out the cosmology and whether we need to copy
read cosmo_idx do_copying <<< "$($codebase/emulator_roles)"

src_dir="/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_${cosmo_idx}"
tmp_dir="/tmp/cosmo_varied_${cosmo_idx}"
finish_marker="${tmp_dir}/FINISHED_COPY"

# the copying takes about 15 minutes
if [ $do_copying -eq 1 ]; then
  echo "$SLURM_TOPOLOGY_ADDR: Started copying $src_dir into $tmp_dir at $(date) ..."
  mkdir -p "$tmp_dir"
  cp -r "${src_dir}/"rockstar_* "$tmp_dir"
  cp "${src_dir}/cosmo.info" "$tmp_dir"
  # convenient for box size
  cp "${src_dir}/fastpm_script.lua" "$tmp_dir"
  echo "$(date)" > $finish_marker
  echo "$SLURM_TOPOLOGY_ADDR: ... Finished copying $src_dir into $tmp_dir at $(date)"
fi

# this may take some time, so increase timeout
utils::wait_for_file $finish_marker $((30 * 60))

function cantor_pairing {
  k1="$1"
  k2="$2"
  a=$(( (k1+k2) * (k1+k2+1) ))
  echo $(( a / 2 + k2 ))
  return 0
}

# unique process id
proc_idx=$(cantor_pairing $SLURM_ARRAY_TASK_ID $OMPI_COMM_WORLD_RANK)

# loop until time is up
consecutive_fails=0

for ii in $( seq 0 10000 ); do
  hod_idx=$(cantor_pairing $proc_idx $ii)

  # we do not consider failure a reason to abort
  bash $codebase/generate_emulator_sample.sh $cosmo_idx $hod_idx \
    && status=$? || status=$?

  if [ $status -ne 0 ]; then
    consecutive_fails=$((consecutive_fails+1))
    utils::printerr "Trial [$cosmo_idx, $hod_idx] failed"
    if [ $consecutive_fails -gt 2 ]; then
      # something must be going terribly wrong
      exit 1
    fi
  else
    consecutive_fails=0
  fi
done
