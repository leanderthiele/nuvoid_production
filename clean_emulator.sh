#!/bin/bash

# NOTE can only run this while emulator sample creation is not taking place!!!

root='/scratch/gpfs/lthiele/nuvoid_production'
for d in "$root"/cosmo_varied_*/emulator/*; do
  rm -f "$d"/lightcone_*.txt &
  rm -rf "$d"/logs_* &
  rm -rf "$d/figs" &
  if ls "$d"/sample_* 1> /dev/null 2>&1; then
    continue
  else
    rm -r "$d" &
  fi
done

wait
