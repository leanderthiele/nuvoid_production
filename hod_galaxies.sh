#!/bin/bash

set -e -o pipefail

# Small wrapper around hod_galaxies.py, forwarding all arguments
# and loading the correct modules

# codes used
HOD_GALAXIES_MODULES="anaconda3/2021.11 gsl/2.6 intel-mkl/2020.1/1/64"
HOD_GALAXIES_CONDA_ENV="galaxies"
HOD_GALAXIES_EXE="python hod_galaxies.py"

module load $HOD_GALAXIES_MODULES
conda activate $HOD_GALAXIES_CONDA_ENV

$HOD_GALAXIES_EXE "$@"

conda deactivate
module rm $HOD_GALAXIES_MODULES
