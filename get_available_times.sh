#!/bin/bash

set -e -o pipefail

# Small helper to figure out which snapshots are available.
# Command line arguments:
#   [1] root directory
#   [2] halo finder (rockstar or rfof)
# Prints out comma-separated list of scale factors

root="$1"
halo_finder="$2"

halo_catalogs=$(ls -d $root/${halo_finder}_*)

# the pattern occuring before the scale factor
if [ "$halo_finder" == "rockstar" ]; then
  pattern='\#a\s=\s'
elif [ "$halo_finder" == "rfof" ]; then
  pattern='Time.*\#HUMANE\s\[\s'
else
  exit 1
fi

out=()
ii=0
for h in $halo_catalogs; do
  if [ "$halo_finder" == "rockstar" ]; then
    header_file="$h/out_${ii}_hosts.bf/Header/attr-v2"
  elif [ "$halo_finder" == "rfof" ]; then
    header_file="$h/Header/attr-v2"
  else
    exit 1
  fi

  a=$(grep -m1 -oP "$pattern"'+\K\d.\d*' "$header_file")
  out+=("$a")
  ii=$((ii+1))
done

echo "$(echo "${out[@]}" | tr ' ' ',')"
