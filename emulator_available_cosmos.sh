#!/bin/bash

# Prints out the available cosmology indices, comma separated,
# as first line,
# then the unavailable ones as second line

avail=()
unavail=()

root="/scratch/gpfs/lthiele/nuvoid_production"

for d in $root/cosmo_varied_*; do
  idx="$(cat $d | tr '_' '\n' | tail -1)"
  ii=0
  for rockstar in $d/rockstar_*; do
    if [ !-d "$rockstar/out_${ii}_hosts.bf" ]; then
      unavail+=($idx)
      break
    fi
  done
  avail+=($idx)
done

echo ${avail[@]} | tr ' ' ','
echo ${unavail[@]} | tr ' ' ','
