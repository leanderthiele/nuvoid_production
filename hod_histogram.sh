#!/bin/bash

set -e -o pipefail

# Command line arguments:
#   [1] vide directory
#   [2] vide output format to use
#   [3] Rmin
#   [4] Rmax
#   [5] Nbins
#   [6] (optional) redshift separators, separated by commas
# Prints out comma separated list of bin counts

codebase="$HOME/nuvoid_production"

# codes used
HOD_HISTOGRAM_MODULES="anaconda3/2021.11"
HOD_HISTOGRAM_EXE="python $codebase/hod_histogram.py"

# command line arguments
vide_dir="$1"
vide_out="$2"
Rmin="$3"
Rmax="$4"
Nbins="$5"
zedges="$6" # may be empty

datafiles=($(ls "$vide_dir/${vide_out}"_centers_central_*.out))
if [ ${#datafiles[@]} -ne 1 ]; then exit 1; fi
datafile="${datafiles[0]}"

module load $HOD_HISTOGRAM_MODULES
out="$($HOD_HISTOGRAM_EXE $datafile $Rmin $Rmax $Nbins $zedges)"
module rm $HOD_HISTOGRAM_MODULES

echo "$out"
