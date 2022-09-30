#!/bin/bash

# Small script to generate fiducial galaxies for testing

source utils.sh
source globals.sh

# codes used in this script
GLX_MODULES="intel-mkl/2020.1/1/64 gsl/2.6 anaconda3/2021.11"
GLX_CONDA_ENV="galaxies"

# this is where the output goes
outdir="$ROOT/galaxies"
mkdir -p $outdir

module load "$GLX_MODULES"
conda activate $GLX_CONDA_ENV

for i in $SNAP_INDICES; do
  python make_galaxies.py $ROOT ${Z_ARR[$i]} $outdir
done


conda deactivate
module rm "$GLX_MODULES"
