#!/bin/bash

set -e -o pipefail

source utils.sh
source globals.sh

# codes used in this script
CLASS_EXE="$HOME/class_public/class"
CLASS_MODULES=" " # module load, rm on empty does not set errno
REPS_EXE="$HOME/reps/reps"
REPS_MODULES="gsl/2.6"

# templates used in this script
CLASS_CFG_TEMPLATE="$HOME/nuvoid_production/find_As.ini"
REPS_CFG_TEMPLATE="$HOME/nuvoid_production/reps.ini"

# check if maybe we don't need to do anything (other process already doing/done it)
if [ -d "$COSMO_WRK_DIR" ]; then
  echo "Cosmology stuff (CLASS and REPS) already computed, skipping"
  utils::wait_for_file "${COSMO_WRK_DIR}/FINISHED"
  exit 0
fi

mkdir -p $COSMO_WRK_DIR

# figure A_s out

class_ini="$cosmo_wrk_dir/find_As.ini"
cp $CLASS_CFG_TEMPLATE $class_ini
utils::replace $class_ini 'Omega_m' "$COSMO_OMEGA_M"
utils::replace $class_ini 'Omega_b' "$COSMO_OMEGA_B"
utils::replace $class_ini 'h'       "$COSMO_HUBBLE"
utils::replace $class_ini 'n_s'     "$COSMO_NS"
utils::replace $class_ini 'sigma8'  "$COSMO_SIGMA8"
utils::replace $class_ini 'N_ur'    "$N_UR"
utils::replace $class_ini 'N_ncdm'  "$COSMO_N_NU"
utils::replace $class_ini 'm_ncdm'  "$COMMA_M_NU"

module load "$CLASS_MODULES"
class_log="$cosmo_wrk_dir/find_As.log"
utils::run "$CLASS_EXE $class_ini" $class_log
module rm "$CLASS_MODULES"

# now we know A_s
A_s=$(grep -m 1 -oP "A\_s\s=\s+\K\d{1}\.\d*e-09" $class_log)

# have some human-readable string that we can store wherever and use to debug anything
# NOTE that I'm not very careful here, individual lines can't have spaces in them...
cosmo_info=$(printf 'hash=%s
                     when=%s
                     #INPUT
                     Omega_m=%.8f
                     Omega_b=%.8f
                     h=%.8f
                     n_s=%.8f
                     sigma_8=%.8f
                     N_nu=%d
                     M_nu=%.8f
                     wrong_nu=%d
                     #DERIVED
                     A_s=%.8e
                     M_nu=%.8f
                     m_nu=%s
                     Omega_nu=%.8f
                     Omega_cdm=%.8f' \
             $cosmo_hash  "$(date +%F@%T)" \
             $COSMO_OMEGA_M $COSMO_OMEGA_B $COSMO_HUBBLE $COSMO_NS $COSMO_SIGMA8 \
             $COSMO_N_NU $COSMO_M_NU $COSMO_WRONG_NU \
             $A_s $M_NU "($COMMA_M_NU)" $OMEGA_NU $OMEGA_CDM)

for s in $cosmo_info; do echo $s >> $COSMO_INFO_FILE; done

# prepare the REPS input file
reps_ini="$COSMO_WRK_DIR/reps.ini"
cp $REPS_CFG_TEMPLATE $reps_ini
utils::replace $reps_ini 'WORKDIR'   "$COSMO_WRK_DIR"
utils::replace $reps_ini 'h'         "$COSMO_HUBBLE"
utils::replace $reps_ini 'OB0'       "$COSMO_OMEGA_B"
utils::replace $reps_ini 'OC0'       "$OMEGA_CDM"
utils::replace $reps_ini 'As'        "$A_s"
utils::replace $reps_ini 'ns'        "$COSMO_NS"
utils::replace $reps_ini 'N_nu'      "$(printf '%.1f' $COSMO_N_NU)"
utils::replace $reps_ini 'wrong_nu'  "$(($COSMO_N_NU==0 ? 0 : $COSMO_WRONG_NU))"
utils::replace $reps_ini 'Neff'      "$N_UR"
utils::replace $reps_ini 'M_nu'      "$M_NU"
utils::replace $reps_ini 'Z_INITIAL' "$(printf '%.1f' $Z_INITIAL)"

# now we get to run REPS
module load "$REPS_MODULES"
utils::run "$REPS_EXE $reps_ini" "$COSMO_WRK_DIR/reps.log"
module rm "$REPS_MODULES"

# if others are waiting for us, we should let them know we are done
# we append so we can diagnose weird race conditions
echo "$(date)" >> "${COSMO_WRK_DIR}/FINISHED"
