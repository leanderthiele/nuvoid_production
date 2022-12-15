#!/bin/bash

# Command line arguments:
#   [1] data directory
#   [2] working directory
#   [3] hod hash
#   [4] augmentation (0..95)
#   [5] halo finder (rockstar or rfof)

# FIXME this is a small hack
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

# how many lightcones we generate per call (with different augmentations)
NSAMPLES=8

set -e -o pipefail

codebase="${HOME}/nuvoid_production"

# codes used
LIGHTCONE_MODULES="gsl/2.6"
LIGHTCONE_EXE="$codebase/lightcone"
SNAP_TIMES_EXE="bash $codebase/get_available_times.sh"

data_dir="$1"
wrk_dir="$2"
hod_hash="$3"
augment="$4"
halo_finder="$5"

# fixed settings
zmin=0.42
zmax=0.70
correct=1
boss_dir="/tigress/lthiele/boss_dr12"
veto=1
stitch_before_RSD=1
verbose=0
binary_output=1

# figure out some stuff
Omega_m=$(grep -m1 -oP 'Omega\_m=+\K\d\.\d*' "$data_dir/cosmo.info")
BoxSize=$(grep -m1 -oP 'boxsize\s=\s+\K\d*(\.\d*)?' "$data_dir/fastpm_script.lua")
comma_snap_times=$($SNAP_TIMES_EXE $data_dir $halo_finder)

# run the code
module load $LIGHTCONE_MODULES

# for some reason we have extremely rare segfaults that are not reproducible,
# so for diagnostic purposes enable core dump
ulimit -c unlimited

# also make sure we don't waste our precious /home quota with core files
# if they do get generated
cd '/scratch/gpfs/lthiele/nuvoid_production/lightcone_core_files'

# we can generate a bunch of samples as this is super cheap
this_augment=$augment
for ii in $( seq 1 $NSAMPLES ); do

  remap=$((this_augment / 48))
  reflecttranslate=$((this_augment % 48))
  $LIGHTCONE_EXE \
    "$wrk_dir/lightcones/$hod_hash" "" \
    $this_augment \
    $BoxSize $Omega_m $zmin $zmax \
    $remap $correct $reflecttranslate \
    $boss_dir $veto $stitch_before_RSD \
    $verbose $binary_output \
    $(echo $comma_snap_times | tr ',' ' ')

  # this works in covering the entire ring, and it gives
  # enough entropy that we shouldn't really worry
  this_augment=$(( (this_augment+37) % 96 ))
done

# and reset to avoid side effects
ulimit -c 0

module rm $LIGHTCONE_MODULES
