#!/bin/bash

# size of the simulation
export BOX_SIZE=2500 # Mpc/h
export NC=2800

# FastPM settings
export Z_INITIAL=99.0
export Z_MID=19.0
export LOG_STEPS=7
export LIN_STEPS=12

# these are only relevant for large neutrino masses,
# as recommended by Bayer+2020
export EARLY_LOG_STEPS=12 # should be divisible by 4
export EARLY_Z_MID=79.0

# where we generate snapshots (order irrelevant, separate by spaces)
export Z_OUT="0.44291191 0.45853203 0.47003594 0.47996333 0.48930041 0.49793992 0.50646687 0.51500717 0.52363417 0.53238288 0.54126877 0.55064664 0.56019118 0.57066313 0.58224299 0.59482622 0.60884719 0.62546052 0.64639842 0.67649879"

# root directory where all the output goes
export BASE="/scratch/gpfs/lthiele/nuvoid_production"

