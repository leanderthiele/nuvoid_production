#!/bin/bash

# Run the parents finder on Rockstar output
# Command line arguments:
# [1] snap index

set -e -o pipefail

source utils.sh
source globals.sh

# codes used in this script
ROCKSTAR_FINDPARENTS_EXE="$HOME/rockstar_bigfile/util/find_parents_bigfile"
ROCKSTAR_MODULES="hdf5/gcc/1.10.0"

snap_idx="$1"
time="${TIMES_ARR[$snap_idx]}"

rockstar_dir="$ROOT/rockstar_$time"

outfile="${rockstar_dir}/out_${snap_idx}.list"
wparents_base="${rockstar_dir}/out_${snap_idx}"

# do nothing if already computed
if [ -d "${wparents_base}_hosts.bf" ]; then
  echo "Not running parents as output already exists"
  exit 0
fi

module load "$ROCKSTAR_MODULES"
utils::run "$ROCKSTAR_FINDPARENTS_EXE $wparents_base $outfile" "$LOGS/rockstar_parents_$time.log"
module rm "$ROCKSTAR_MODULES"

# if we got here and have successfully generated the output, we can delete the intermediate file
if [ -d "${wparents_base}_hosts.bf" ]; then
  rm $outfile
fi