#!/bin/bash

samples_idx="$1"
time_samples="$2"

# some useful calculations
source globals.sh

augment=$SLURM_PROCID

module load gsl/2.6

./lightcone \
  "/scratch/gpfs/lthiele/nuvoid_production/test1" \
  "fidhod" \
  "time_samples_${samples_idx}_augment${augment}_remap${REMAP_CASE}" \
  $BOX_SIZE \
  0.30 \
  0.42 0.70 \
  $REMAP_CASE \
  $CORRECT \
  $augment \
  "/tigress/lthiele/boss_dr12" \
  $VETO \
  $time_samples
