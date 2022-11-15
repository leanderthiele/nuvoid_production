#!/bin/bash

set -e -o pipefail

# Clean up after computations to save space
# Command line arguments:
#   [1] working directory
#   [2] hod hash
#   [3] step (0 for .bin files, 1 for rest)

wrk_dir="$1"
hod_hash="$2"
step="$3"

if [ $step -eq 0 ]; then
  # first pass (lightcone has been generated but vide not been run)
  rm "$wrk_dir/emulator/$hod_hash/"galaxies_*.bin
else
  # second pass (vide has been run)
  rm "$wrk_dir/emulator/$hod_hash/"lightcone_*.txt

  vide_log_dirs="$(ls -d "$wrk_dir/emulator/$hod_hash/"logs_*)"
  vide_fig_dir="$wrk_dir/emulator/$hod_hash/figs"
  vide_out_dirs="$(ls -d "$wrk_dir/emulator/$hod_hash/"sample_*)"

  rm -r $vide_log_dirs
  rm -r $vide_fig_dir

  for vide_out_dir in $vide_out_dirs; do
    find $vide_out_dir -type f ! -name '*.out' -delete
  done
fi
