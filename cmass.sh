#!/bin/bash

# Contains the settings that are always needed when running for a comparison with Quijote


# size of the simulation
export BOX_SIZE=2500 # Mpc/h

# FastPM settings
export Z_INITIAL=99.0
export Z_MID=19.0
export LOG_STEPS=5
export LIN_STEPS=10

# where we generate snapshots (order irrelevant, separate by spaces)
export Z_OUT="0.5 0.55 0.6 0.65 0.7"

# root directory where all the output goes
export BASE="/scratch/gpfs/lthiele/nuvoid_production"

