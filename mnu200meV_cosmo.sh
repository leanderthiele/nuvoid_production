#!/bin/bash

# this is the fiducial Quijote cosmology
export COSMO_OMEGA_M=0.3175 # includes any neutrinos
export COSMO_OMEGA_B=0.0490
export COSMO_HUBBLE=0.6711
export COSMO_NS=0.9624
export COSMO_SIGMA8=0.834

# must be integer, only 0 and 3 supported so far
export COSMO_N_NU=3

# neutrino mass sum, has no influence if N_NU=0
export COSMO_M_NU=0.2

# REPS setting, only applies when N_NU>0
export COSMO_WRONG_NU=1
