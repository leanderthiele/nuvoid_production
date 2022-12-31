#!/bin/bash

# Command line arguments:
#   [1] input directory (where the lightcone_[0-9]*.txt files may be found)
#       The input directory will *not* be cleaned up so it should be on /tmp
#   [2] output directory (where we copy the voids_[0-9]* folders upon success)

set -e -o pipefail

codebase="$HOME/nuvoid_production"

source $codebase/utils.sh

# codes used
VIDE_MODULES="anaconda3/2021.11"
VIDE_CONDA_ENV="galaxies"
VIDE_EXE="python -u -m void_pipeline"

# templates used
VIDE_CFG_TEMPLATE="$codebase/lightcones_vide_cfg.py"

# command line arguments
indir="$1"
outdir="$2"

# some fixed settings
zmin=0.42
zmax=0.70
Omega_m=0.3439 # mean of our prior, also consistent with Plk BOSS analysis

logdir="$indir/logs"
figdir="$indir/figs"
mkdir -p "$logdir" "$figdir"
vide_cfg="$indir/vide_cfg.py"

lightcones=("$indir"/lightcone_[0-9]*.txt) # this is an array
num_lightcones="${#lightcones[@]}"
if [ $num_lightcones -eq 0 ]; then
  exit 1
fi


cd $logdir
module load $VIDE_MODULES
conda activate $VIDE_CONDA_ENV

num_success=0

for lightcone in "${lightcones[@]}"; do

  augment="$(echo $lightcone | grep -m 1 -oP 'lightcone_+\K\d*')"
  retry_index=0

  cp $VIDE_CFG_TEMPLATE $vide_cfg
  utils::replace $vide_cfg 'inputDataDir'    "$indir"
  utils::replace $vide_cfg 'workDir'         "$indir"
  utils::replace $vide_cfg 'logDir'          "$logdir"
  utils::replace $vide_cfg 'figDir'          "$figdir"
  utils::replace $vide_cfg 'zmin'            "$zmin"
  utils::replace $vide_cfg 'zmax'            "$zmax"
  utils::replace $vide_cfg 'augment'         "$augment"
  utils::replace $vide_cfg 'Omega_m'         "$Omega_m"
  utils::replace $vide_cfg 'retry_index'     "$retry_index"
  utils::replace $vide_cfg 'numZobovThreads' "$OMP_NUM_THREADS"

  vide_log=$indir/vide_$augment.log
  utils::run "$VIDE_EXE $vide_cfg" "$vide_log" \
    && status=$? || status=$?

  if [ $status -ne 0 ]; then
    # occasional VIDE failure is acceptable, it happens, we just go to the next lightcone
    utils::printerr "VIDE failed for $lightcone"
    continue
  fi

  num_success=$((num_success+1))

  # copy output into permanent storage
  tmp_out="$indir/sample_$augment"
  perm_out="$outdir/voids_$augment"
  mkdir -p $perm_out
  cp "$tmp_out"/*.out $perm_out

  # make sure we don't run out of disk space
  rm -r "$tmp_out"

done

conda deactivate
module rm $VIDE_MODULES

if [ $num_success -ne $num_lightcones ]; then
  utils::printerr "failures: succeeded $num_success/$num_lightcones in $outdir"
fi

if [ $num_success -eq 0 ]; then
  exit 1
fi
