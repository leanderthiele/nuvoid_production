#!/bin/bash

# Command line arguments:
#   [1] input directory (where the lightcone_[0-9]*.txt files may be found)
#       The input directory will *not* be cleaned up so it should be on /tmp
#   [2] output directory (where we copy the voids_[0-9]* folders upon success)
#   [3] lightcone index

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
lightcone_idx="$3"

# some fixed settings
zmin=0.42
zmax=0.70
Omega_m=0.3439 # mean of our prior, also consistent with Plk BOSS analysis

logdir="$indir/logs_${lightcone_idx}"
figdir="$indir/figs_${lightcone_idx}"
mkdir -p "$logdir" "$figdir"
vide_cfg="$indir/vide_cfg_${lightcone_idx}.py"

cd $logdir
module load $VIDE_MODULES
conda activate $VIDE_CONDA_ENV

vide_max_tries=8
vide_failed=1
for retry_index in $( seq 0 $((vide_max_tries-1)) ); do
  cp $VIDE_CFG_TEMPLATE $vide_cfg
  utils::replace $vide_cfg 'inputDataDir'    "$indir"
  utils::replace $vide_cfg 'workDir'         "$indir"
  utils::replace $vide_cfg 'logDir'          "$logdir"
  utils::replace $vide_cfg 'figDir'          "$figdir"
  utils::replace $vide_cfg 'zmin'            "$zmin"
  utils::replace $vide_cfg 'zmax'            "$zmax"
  utils::replace $vide_cfg 'augment'         "$lightcone_idx"
  utils::replace $vide_cfg 'Omega_m'         "$Omega_m"
  utils::replace $vide_cfg 'retry_index'     "$retry_index"
  utils::replace $vide_cfg 'numZobovThreads' "$OMP_NUM_THREADS"

  vide_log=$indir/vide_$lightcone_idx.log
  utils::run "$VIDE_EXE $vide_cfg" "$vide_log" \
    && status=$? || status=$?
  
  if [ $status -eq 0 ]; then
    vide_failed=0
    break
  fi

  # ok, VIDE failed. We need to make sure it did so in the expected way
  # (not out of disk space or something)
  if [ "$(tail -1 $vide_log)" != '  Extracting voids with ZOBOV... FAILED!' ]; then
    utils::printerr "VIDE failed in an unexpted way for $wrk_dir-$hod_hash-$augment"
    exit 43
  fi
done

if [ $vide_failed -eq 1 ]; then
    utils::printerr "VIDE did not succeed within $vide_max_tries tries for $indir-$lightcone_idx"
    exit 42
fi

# copy output into permanent storage
tmp_out="$indir/sample_${lightcone_idx}"
perm_out="$outdir/voids_${lightcone_idx}"
mkdir -p $perm_out
cp "$tmp_out"/*.out $perm_out

# make sure we don't run out of disk space
rm -r "$tmp_out"

conda deactivate
module rm $VIDE_MODULES
