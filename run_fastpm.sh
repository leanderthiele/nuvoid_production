#!/bin/bash

# Driver code to run FastPM

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/utils.sh
source $codebase/globals.sh

# codes used in this script
TIMESTEPS_EXE="$codebase/timesteps"
FASTPM_EXE="$HOME/fastpm_allintel/src/fastpm"
FASTPM_MODULES="anaconda3/2021.11 intel-mpi/intel/2019.7/64 gsl/2.6"
FASTPM_CONDA_ENV="fastpm"

# templates used in this script
FASTPM_CFG_TEMPLATE="$codebase/fastpm_script.lua"

# only run FastPM if output not existent
if [ -d "$ROOT/snap_${TIMES_ARR[$((NUM_SNAPS-1))]}" ]; then
  echo "Not running FastPM as output already exists"
  exit 0
fi

cp $COSMO_INFO_FILE "$ROOT/"

mkdir -p "$ROOT/powerspectra"

# hash the ID if seed not defined
if [ -z $SEED ]; then SEED=$(utils::dec_hash "$ID" 32); fi

# compute the FastPM time steps
# as recommended by Bayer+2020, we take a bunch of early logarithmic steps
# if the neutrino mass is quite large
if [ $(utils::feval "$COSMO_M_NU < 0.1" '%d') -eq 1 ]; then
  early_steps=0
elif [ $(utils::feval "$COSMO_M_NU < 0.2" '%d') -eq 1 ]; then
  early_steps=$((EARLY_LOG_STEPS / 4))
elif [ $(utils::feval "$COSMO_M_NU < 0.4" '%d') -eq 1 ]; then
  early_steps=$((EARLY_LOG_STEPS / 2))
else
  early_steps=$EARLY_LOG_STEPS
fi
time_steps="$($TIMESTEPS_EXE $Z_INITIAL $Z_MID $Z_EARLY $LOG_STEPS $LIN_STEPS $early_steps $NUM_SNAPS $TIMES)"

# write our input file
fastpm_cfg="$ROOT/fastpm_script.lua"
cp $FASTPM_CFG_TEMPLATE $fastpm_cfg
utils::replace $fastpm_cfg 'nc'            "$NC"
utils::replace $fastpm_cfg 'boxsize'       "$BOX_SIZE"
utils::replace $fastpm_cfg 'random_seed'   "$SEED"
utils::replace $fastpm_cfg 'FASTPM_OUTPUT' "$ROOT"
utils::replace $fastpm_cfg 'OUT_REDSHIFTS' "$(echo "${Z_ARR[@]}" | tr " " ",")"
utils::replace $fastpm_cfg 'Omega_m'       "$COSMO_OMEGA_M"
utils::replace $fastpm_cfg 'h'             "$COSMO_HUBBLE"
utils::replace $fastpm_cfg 'REPS_OUTPUT'   "$COSMO_WRK_DIR"
utils::replace $fastpm_cfg 'TIME_STEPS'    "$time_steps"
utils::replace $fastpm_cfg 'Z_INITIAL'     "$(printf '%.4f' $Z_INITIAL)"

# for the FastPM file, we avoid confusion by removing neutrino part entirely if not necessary
if [ "$COSMO_N_NU" -eq "0" ]; then
  sed -i '/STARTNEUTRINOS/,/ENDNEUTRINOS/d' $fastpm_cfg
else
  # TODO not sure about Neff
  utils::replace $fastpm_cfg 'N_eff' "3.046"
  utils::replace $fastpm_cfg 'N_nu'  "$COSMO_N_NU"
  utils::replace $fastpm_cfg 'm_nu'  "$COMMA_M_NU"
fi

# FIXME
exit 1

# no idea what these do
export OMPI_MCA_rmaps_base_no_oversubscribe=0
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_mpi_yield_when_idle=1
export OMPI_MCA_mpi_show_mca_params=1

# for the FastPM call
export SRUN_CPUS_PER_TASK=1
export OMP_NUM_THREADS=1

# prepare environment for FastPM
module load "$FASTPM_MODULES"
conda activate "$FASTPM_CONDA_ENV"

# have checked that -W and -f flags don't make a difference
utils::run "mpirun -n $SLURM_NTASKS $FASTPM_EXE -T $OMP_NUM_THREADS $fastpm_cfg" "$LOGS/fastpm.log"

conda deactivate
module rm "$FASTPM_MODULES"
