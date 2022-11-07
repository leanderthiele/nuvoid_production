#!/bin/bash

# Cosmology template to be overwritten and sourced

export ID="cosmo_fiducial_<<<idx>>>"
# seed will be fixed by hashing the ID

# these are chosen as the means of our priors
# given the constraint 0.06<M_nu<0.15
# This is done approximately using our set of the first 256
# quasi-random samples. The mentioned selection then includes
# 39 samples which should be enough for fairly robust means.
export COSMO_OMEGA_M='0.3219034692307693'
export COSMO_OMEGA_B='0.049976779230769236'
export COSMO_HUBBLE='0.66908459'
export COSMO_NS='0.9651028379487179'

# A_s with mean over log (delta~0.001 compared to linear mean)
export COSMO_AS='2.0971604141485533e-09'

# seems reasonable
export COSMO_M_NU='0.1'

# fixed
export COSMO_WRONG_NU=1
export COSMO_N_NU=3
export COSMO_TAU=0.0544
