#!/bin/bash

# NOTE can only run this while emulator sample creation is not taking place!!!

root='/scratch/gpfs/lthiele/nuvoid_production'
for d in "$root"/cosmo_varied_*/emulator/*; do
  rm -f "$d"/lightcone_*.txt &
  rm -rf "$d"/logs_* &
  rm -rf "$d/figs" &

  # the following file is the last one VIDE generates
  if ls "$d"/sample_*/trimmed_nodencut_sky_positions_all_*.out 1> /dev/null 2>&1; then
    continue
  else
    rm -r "$d" &
  fi
done

wait
