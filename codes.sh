#!/bin/bash

# Contains the executables and the environments they live in

# =========== EXECUTABLES ============
CLASS_EXE="$HOME/class_public/class"
REPS_EXE="$HOME/reps/reps"
FASTPM_EXE="$HOME/fastpm_allintel/src/fastpm"
ROCKSTAR_EXE="$HOME/rockstar_bigfile/rockstar"
ROCKSTAR_DRIVER_EXE="bash ./rockstar_driver.sh"
ROCKSTAR_FINDPARENTS_EXE="$HOME/rockstar_bigfile/util/find_parents_bigfile"
ROCKSTAR_BALANCE="./rockstar_balance"
ZOBOV_ROOT="/tigress/lthiele/.conda/envs/galaxies/lib/python3.10/site-packages/vide-2.0-py3.10-linux-x86_64.egg/vide/bin"
VIDE_EXE="python -u -m void_pipeline"

# ============ REQUIRED ENVIRONMENTS ===========
CLASS_MODULES=" " # module load, rm on empty does not set errno
REPS_MODULES="gsl/2.6"
FASTPM_MODULES="anaconda3/2021.11 intel-mpi/intel/2019.7/64 gsl/2.6"
FASTPM_CONDA_ENV="fastpm"
ROCKSTAR_MODULES="hdf5/gcc/1.10.0"
GLX_MODULES="intel-mkl/2020.1/1/64 gsl/2.6 anaconda3/2021.11"
GLX_CONDA_ENV="galaxies"
VIDE_MODULES="anaconda3/2021.11"
VIDE_CONDA_ENV="galaxies"
