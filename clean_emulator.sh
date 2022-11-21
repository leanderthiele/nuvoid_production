#!/bin/bash

# NOTE this also works while jobs are running now!
# Pass any argument to skip the check for old file

skip_check="$1"

if [ ! -z $skip_check ]; then
  echo "Warning: not checking for old file. Dangerous while jobs are running!"
fi

now_time=$(date +'%s')

# let at least 2 hours be elapsed since creation so we know
# for sure nobody is working on this right now
min_time_delta=$(( 2 * 60 * 60 ))

root='/scratch/gpfs/lthiele/nuvoid_production'
for d in "$root"/cosmo_varied_*/emulator/*; do
  # check for stale/work in progress
  if [ -z $skip_check ]; then
    if [ ! -f $d/hod.info ]; then
      continue
    fi

    create_time=$(date +'%s' -r $d/hod.info)

    # if the directory is relatively recent, do not touch
    # (someone may be working on it)
    if [ $(( now_time-create_time )) -lt $min_time_delta ]; then
      continue
    fi
  fi

  # the following file is the last one VIDE generates
  if ls "$d"/sample_*/trimmed_nodencut_sky_positions_all_*.out 1> /dev/null 2>&1; then
    rm -f "$d"/lightcone_*.txt &
    rm -rf "$d"/logs_* &
    rm -rf "$d/figs" &
  else
    rm -r "$d" &
  fi
done

wait
