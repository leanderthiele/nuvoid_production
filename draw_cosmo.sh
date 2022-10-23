#!/bin/bash

# Command line arguments:
#  [1] index in the quasi-random sequence
#
# Prints out comma separated list of our 6-parameter model,
#    Omega_B, Omega_M, h0, A_s, n_s, M_nu

codebase=$HOME/nuvoid_production

# codes to use
SAMPLE_PRIOR_MODULES="gsl/2.6"
SAMPLE_PRIOR_EXE="$HOME/nuvoid_production/sample_prior"
REPARAMATERIZE_MODULES="anaconda3/2021.11"
REPARAMATERIZE_CONDA_ENV="galaxies"
REPARAMATERIZE_EXE="python $codebase/reparameterize.py"

# priors
GAUSS_PRIOR_DIM=5
GAUSS_PRIOR_FILE="$codebase/mu_cov_plikHM_TTTEEE_lowl_lowE.dat"
UNIFORM_PRIOR_DIM=1
UNIFORM_PRIOR_FILE="$codebase/mnu_prior.dat"

# the index
IDX="$1"

# draw the CMB parameterization
module load "$SAMPLE_PRIOR_MODULES"
cmb_params="$($SAMPLE_PRIOR_EXE $IDX $GAUSS_PRIOR_DIM $GAUSS_PRIOR_FILE $UNIFORM_PRIOR_DIM $UNIFORM_PRIOR_FILE)"
module rm "$SAMPLE_PRIOR_MODULES"

# convert into our parameterization
module load "$REPARAMATERIZE_MODULES"
conda activate $REPARAMATERIZE_CONDA_ENV
our_params="$($REPARAMATERIZE_EXE $(echo $cmb_params | tr ',' ' '))"
conda deactivate
module rm "$REPARAMATERIZE_MODULES"

echo $our_params
