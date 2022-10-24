#!/bin/bash

set -e -o pipefail

codebase=$HOME/nuvoid_production

source $codebase/utils.sh
source $codebase/globals.sh

# codes used in this script
CLASS_EXE="$HOME/class_public/class"
CLASS_MODULES=" " # module load, rm on empty does not set errno
REPS_EXE="$HOME/reps/reps"
REPS_MODULES="gsl/2.6"

# templates used in this script
CLASS_CFG_AS_TEMPLATE="$codebase/find_As.ini"
CLASS_CFG_S8_TEMPLATE="$codebase/find_sigma8.ini"
REPS_CFG_TEMPLATE="$codebase/reps.ini"

# check if maybe we don't need to do anything (other process already doing/done it)
if [ -d "$COSMO_WRK_DIR" ]; then
  echo "Cosmology stuff (CLASS and REPS) already computed, skipping"
  utils::wait_for_file "${COSMO_WRK_DIR}/FINISHED"
  exit 0
fi

mkdir -p $COSMO_WRK_DIR

function class_ini_common {
  ini="$1"
  output_dir="${COSMO_WRK_DIR}/class_output"
  mkdir -p $output_dir

  utils::replace $ini 'Omega_m'  "$COSMO_OMEGA_M"
  utils::replace $ini 'Omega_b'  "$COSMO_OMEGA_B"
  utils::replace $ini 'h'        "$COSMO_HUBBLE"
  utils::replace $ini 'n_s'      "$COSMO_NS"
  utils::replace $ini 'N_ur'     "$N_UR"
  utils::replace $ini 'N_ncdm'   "$COSMO_N_NU"
  utils::replace $ini 'm_ncdm'   "$COMMA_M_NU"
  utils::replace $ini 'tau_reio' "$COSMO_TAU"
  utils::replace $ini 'output'   "$output_dir"

  return 0
}

if [ -z $COSMO_AS ]; then
  # figure A_s out

  class_ini="$COSMO_WRK_DIR/find_As.ini"
  cp $CLASS_CFG_AS_TEMPLATE $class_ini
  class_ini_common $class_ini
  utils::replace $class_ini 'sigma8' "$COSMO_SIGMA8"

  module load "$CLASS_MODULES"
  class_log="$COSMO_WRK_DIR/find_As.log"
  utils::run "$CLASS_EXE $class_ini" $class_log
  module rm "$CLASS_MODULES"

  # now we know A_s
  COSMO_AS=$(grep -m 1 -oP "A\_s\s=\s+\K\d{1}\.\d*e-09" $class_log)
else
  # figure sigma_8 out
  
  class_ini="$COSMO_WRK_DIR/find_sigma8.ini"
  cp $CLASS_CFG_S8_TEMPLATE $class_ini
  class_ini_common $class_ini
  utils::replace $class_ini 'A_s' "$COSMO_AS"

  module load "$CLASS_MODULES"
  class_log="$COSMO_WRK_DIR/find_sigma8.log"
  utils::run "$CLASS_EXE $class_ini" $class_log
  module rm "$CLASS_MODULES"

  # NOTE that the first sigma8 printout is for total matter so we are good
  COSMO_SIGMA8=$(grep -m 1 -oP "sigma8=+\K\d{1}\.\d*" $class_log)
fi

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
             $COSMO_HASH  "$(date +%F@%T)" \
             $COSMO_OMEGA_M $COSMO_OMEGA_B $COSMO_HUBBLE $COSMO_NS $COSMO_SIGMA8 \
             $COSMO_N_NU $COSMO_M_NU $COSMO_WRONG_NU \
             $COSMO_AS $M_NU "($COMMA_M_NU)" $OMEGA_NU $OMEGA_CDM)

for s in $cosmo_info; do echo $s >> $COSMO_INFO_FILE; done

# prepare the REPS input file
reps_ini="$COSMO_WRK_DIR/reps.ini"
cp $REPS_CFG_TEMPLATE $reps_ini
utils::replace $reps_ini 'WORKDIR'   "$COSMO_WRK_DIR"
utils::replace $reps_ini 'h'         "$COSMO_HUBBLE"
utils::replace $reps_ini 'OB0'       "$COSMO_OMEGA_B"
utils::replace $reps_ini 'OC0'       "$OMEGA_CDM"
utils::replace $reps_ini 'As'        "$COSMO_AS"
utils::replace $reps_ini 'ns'        "$COSMO_NS"
utils::replace $reps_ini 'N_nu'      "$(printf '%.1f' $COSMO_N_NU)"
utils::replace $reps_ini 'wrong_nu'  "$(($COSMO_N_NU==0 ? 0 : $COSMO_WRONG_NU))"
utils::replace $reps_ini 'Neff'      "$N_UR"
utils::replace $reps_ini 'M_nu'      "$M_NU"
utils::replace $reps_ini 'tau_reio'  "$COSMO_TAU"
utils::replace $reps_ini 'Z_INITIAL' "$(printf '%.1f' $Z_INITIAL)"

# now we get to run REPS
module load "$REPS_MODULES"
utils::run "$REPS_EXE $reps_ini" "$COSMO_WRK_DIR/reps.log"
module rm "$REPS_MODULES"

# if others are waiting for us, we should let them know we are done
# we append so we can diagnose weird race conditions
echo "$(date)" >> "${COSMO_WRK_DIR}/FINISHED"
