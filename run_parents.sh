#!/bin/bash

# Run the parents finder on Rockstar output
# Command line arguments:
# [1] snap index

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/utils.sh
source $codebase/globals.sh

# codes used in this script
ROCKSTAR_FINDPARENTS_EXE="$HOME/rockstar_bigfile/util/find_parents_bigfile"
ROCKSTAR_MODULES="hdf5/gcc/1.10.0"

snap_idx="$1"
time="${TIMES_ARR[$snap_idx]}"

rockstar_dir="$ROOT/rockstar_$time"

outfile="${rockstar_dir}/out_${snap_idx}.list"
wparents_base="${rockstar_dir}/out_${snap_idx}"
rockstar_finished_file="${rockstar_dir}/FINISHED_ROCKSTAR"
parents_finished_file="${rockstar_dir}/FINISHED_PARENTS"

# do nothing if already computed
# if [ -d "${wparents_base}_hosts.bf" ]; then
#   echo "Not running parents as output already exists"
#   exit 0
# fi

# do nothing if no input available
# if [ ! -f $outfile ]; then
#   echo "Not running parents as input does not exist"
#   exit 0
# fi
if [ -f "$parents_finished_file" ]; then
  echo "Not running parents as $parents_finished_file already exists"
  exit 0
fi

if [ ! -f "$rockstar_finished_file" ]; then
  utils::printerr "Rockstar not done, $rockstar_finished_file does not exist!"
  exit 1
fi

module load "$ROCKSTAR_MODULES"
utils::run "$ROCKSTAR_FINDPARENTS_EXE $wparents_base $outfile" "$LOGS/rockstar_parents_$time.log"
module rm "$ROCKSTAR_MODULES"

# if we got here and have successfully generated the output, we can delete the intermediate file
if [ -d "${wparents_base}_hosts.bf" ]; then
  rm $outfile
fi

# we are done
echo "$(date)" > "$parents_finished_file"
