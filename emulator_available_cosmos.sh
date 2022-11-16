#!/bin/bash

set -e -o pipefail

# Prints out the available cosmology indices, comma separated,
# as first line,
# then the unavailable ones as second line

avail=()
unavail=(-1)

root="/scratch/gpfs/lthiele/nuvoid_production"

for d in $root/cosmo_varied_*; do
  idx="$(echo $d | tr '_' '\n' | tail -1)"
  ii=0
  not_found=0
  for rockstar in $d/rockstar_*; do
    if [ ! -d "$rockstar/out_${ii}_hosts.bf" ]; then
      unavail+=($idx)
      not_found=1
      break
    fi
    ii=$((ii+1))
  done
  if [ $not_found -eq 0 ]; then
    avail+=($idx)
  fi
done

echo "$(echo -n "$(echo "${avail[@]}" | tr ' ' '\n' | sort -h)" | tr '\n' ',')"
echo "$(echo -n "$(echo "${unavail[@]}" | tr ' ' '\n' | sort -h)" | tr '\n' ',')"
