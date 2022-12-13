#!/bin/bash

set -e -o pipefail

# Clean up after computations to save space
# Command line arguments:
#   [1] working directory
#   [2] hod hash

wrk_dir="$1"
hod_hash="$2"

rm "$wrk_dir/lightcones/$hod_hash/"galaxies_*.bin
