#!/bin/bash

set -e -o pipefail

# Clean up after computations to save space
# Command line arguments:
#   [1] working directory
#   [2] hod hash

wrk_dir="$1"
hod_hash="$2"

rm "$wrk_dir/hod/$hod_hash/"galaxies_*.bin
rm "$wrk_dir/hod/$hod_hash/"lightcone_*.txt

vide_log_dirs="$(ls -d "$wrk_dir/hod/$hod_hash/"logs_*)"
vide_fig_dir="$wrk_dir/hod/$hod_hash/figs"
vide_out_dirs="$(ls -d "$wrk_dir/hod/$hod_hash/"sample_*)"

rm -r $vide_log_dirs
rm -r $vide_fig_dir

for vide_out_dir in $vide_out_dirs; do
  find $vide_out_dir -type f ! -name '*.out' -delete
done
