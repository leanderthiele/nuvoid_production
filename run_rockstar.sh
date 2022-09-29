#!/bin/bash

# Runs Rockstar for a single snapshot, will fill an entire slurm job
# Command line arguments:
# [1] time index (everything else is exported)

set -e -o pipefail

source utils.sh
source globals.sh

# codes used in this script
ROCKSTAR_EXE="$HOME/rockstar_bigfile/rockstar"
ROCKSTAR_MODULES="hdf5/gcc/1.10.0"

# templates used in this script
ROCKSTAR_CFG_TEMPLATE="$HOME/nuvoid_production/rockstar_server.cfg"


snap_idx="$1"
time=${TIMES_ARR[$snap_idx]}
snap_dir="$ROOT/snap_$time"
rockstar_dir="$ROOT/rockstar_$time"
snap_link="${rockstar_dir}/snap_${snap_idx}"

# only run if output does not exist
if [ -f "$rockstar_dir/out_${snap_idx}.list" ]; then
  echo "Not running Rockstar as .list output already exists"
  exit 0
fi

if [ -d "$rockstar_dir/out_${snap_idx}_hosts.bf" ]; then
  echo "Not running Rockstar as .bf output already exists"
  exit 0
fi

# check if we actually have the data available
if [ ! -d $snap_dir ]; then
  echo "Not running Rockstar as snapshot does not exist"
  exit 0
fi

mkdir -p $rockstar_dir
rm -f $snap_link
ln -s $snap_dir $snap_link

export SRUN_CPUS_PER_TASK=40
export OMP_NUM_THREADS=40

# find out how many blocks the FastPM bigfile is split into
block_pattern=$(for i in {1..6}; do printf "[0-9,A-F]"; done)
num_blocks=$(find ${snap_dir}/1/ID/${block_pattern} | wc -l)

# generate the input rockstar file
num_cpus=$((40*SLURM_JOB_NUM_NODES))
rockstar_cfg="${rockstar_dir}/rockstar_server_${time}.cfg"
cp $ROCKSTAR_CFG_TEMPLATE $rockstar_cfg
utils::replace $rockstar_cfg 'INBASE'        "$rockstar_dir"
utils::replace $rockstar_cfg 'OUTBASE'       "$rockstar_dir"
utils::replace $rockstar_cfg 'NUM_BLOCKS'    "$num_blocks"
utils::replace $rockstar_cfg 'NUM_SNAPS'     "$((snap_idx+1))"
utils::replace $rockstar_cfg 'STARTING_SNAP' "$snap_idx"
utils::replace $rockstar_cfg 'NUM_READERS'   "$((num_blocks < num_cpus ? num_blocks : num_cpus))"
utils::replace $rockstar_cfg 'NUM_WRITERS'   "$num_cpus"
utils::replace $rockstar_cfg 'FORK_PROCESSORS_PER_MACHINE' 40

module load "$ROCKSTAR_MODULES"

# start the server
utils::run "$ROCKSTAR_EXE -c $rockstar_cfg" "$LOGS/rockstar_server_${time}.log" &

# wait for configuration file to be generated
auto_cfg="${rockstar_dir}/auto-rockstar.cfg"
utils::wait_for_file $auto_cfg

# file has appeared, let's rock
srun_cmd="srun -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 -W 0"
utils::run "$srun_cmd $ROCKSTAR_EXE -c $auto_cfg" "$LOGS/rockstar_workers_${time}.log"

# clean up
module rm "$ROCKSTAR_MODULES"

# if we have finished successfully, we should delete the snapshot
if [ -f "$rockstar_dir/out_${snap_idx}.list" ]; then
  rm -f $snap_link &
  rm -rf $snap_dir &
  rm -f $rockstar_dir/halos_* &
  rm -rf $rockstar_dir/profiling &
  wait
fi
