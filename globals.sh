#!/bin/bash

# This is to be sourced by other codes, it simply provides some initial settings and calculations

set -e -o pipefail

source utils.sh

ROOT="$BASE/$ID"
COSMO_ROOT="$BASE/cosmologies"
LOGS="$ROOT/logs"
mkdir -p $ROOT $COSMO_ROOT $LOGS

# ============ PROCESS THE GIVEN REDSHIFTS ===============

# the snapshot scale factors, in increasing order and the FastPM conventional string format
Z_ARR=($Z_OUT)
TIMES=""
for z in "${Z_ARR[@]}"; do TIMES="$TIMES $(utils::feval "1./(1.+$z)" '%.4f')"; done
TIMES="$(echo "$TIMES" | tr " " "\n" | sort | tr "\n" " ")"

# get them also in array format which makes life easier
TIMES_ARR=($TIMES)
NUM_SNAPS=${#TIMES_ARR[@]}
SNAP_INDICES="$(seq 0 $((NUM_SNAPS-1)))"

# ============ FIGURE SOME BACKGROUND COSMOLOGY OUT =============

# recommended values from CLASS explanatory.ini, indices corresponding to N_nu
N_ur_arr=(3.044 2.0308 1.0176 0.00441)

# get ours
N_UR=${N_ur_arr[$COSMO_N_NU]}

# holds the individual masses, which at the moment are assumed equal
# (empty if COSMO_N_NU=0)
M_NU_ARR=()
for i in $(seq 1 "$COSMO_N_NU"); do M_NU_ARR+=("$(utils::feval "$COSMO_M_NU / $COSMO_N_NU")"); done

# list of M_NU_ARR separated by commas
COMMA_M_NU=$(echo "${M_NU_ARR[@]}" | tr " " ",")

# neutrino mass sum, Omega_nu
if [ "$COSMO_N_NU" -eq "0" ]; then
  M_NU="0.0"
  OMEGA_NU="0.0"
else
  # we compute the mass sum not just as COSMO_M_NU but rather as the sum
  # of the m_nu so we only have to change the code in one place if we
  # want to change the mass splitting
  M_NU=$(utils::feval "$(echo ${M_NU_ARR[@]} | tr " " "+")" '%.16f')
  OMEGA_NU=$(utils::feval "$M_NU / ($COSMO_HUBBLE^2 * 93.13306)" '%.16f')
fi

OMEGA_CDM=$(utils::feval "$COSMO_OMEGA_M - $COSMO_OMEGA_B - $OMEGA_NU" '%.16f')


# ============ HASH OUR COSMOLOGY AND POINT OUTPUT DIRECTORY TO IT =================

# in order to avoid duplicates which are just source of error, we hash our cosmology
if [ -z $COSMO_AS ]
  amplitude=$COSMO_SIGMA8
else
  # since we use .8f below we need to shift
  amplitude=$(utils::feval "$COSMO_AS * 1e9" '%.16f')
COSMO_HASH_STR="$(printf '%.8f %.8f %.8f %.8f %.8f %d' \
                  $COSMO_OMEGA_M $COSMO_OMEGA_B $COSMO_HUBBLE $COSMO_NS $amplitude $COSMO_N_NU)"

if [ "$COSMO_N_NU" -gt "0" ]; then
  COSMO_HASH_STR="$COSMO_HASH_STR $(printf '%.8f %d' $COSMO_M_NU $COSMO_WRONG_NU)"
fi

# create a hexadecimal hash
COSMO_HASH=$(utils::hex_hash "$COSMO_HASH_STR")
COSMO_WRK_DIR="$COSMO_ROOT/$COSMO_HASH"
# NOTE we do not create this directory yet since we consider existence of it an indicator

# for reference
COSMO_INFO_FILE="$COSMO_WRK_DIR/cosmo.info"
