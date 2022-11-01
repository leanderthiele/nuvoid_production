#!/bin/bash

# Command line arguments:
#  [1] working directory
#  [2] hod hash
#  [3] augmentation index

set -e -o pipefail

codebase="$HOME/nuvoid_production"

source $codebase/utils.sh

# codes used
VIDE_MODULES="anaconda3/2021.11"
VIDE_CONDA_ENV="galaxies"
VIDE_EXE="python -u -m vide_pipeline"

# templates used
VIDE_CFG_TEMPLATE="$codebase/hod_vide_cfg.py"

# command line arguments
wrk_dir="$1"
hod_hash="$2"
augment="$3"

# some fixed settings
zmin=0.42
zmax=0.70
Omega_m=0.30 # TODO

logdir="$wrk_dir/hod/$hod_hash/logs_$augment"
figdir="$wrk_dir/hod/$hod_hash/figs"
mkdir -p "$logdir" "$figdir"
vide_cfg="$wrk_dir/hod/$hod_hash/vide_cfg_$augment.py"

cp $VIDE_CFG_TEMPLATE $vide_cfg
utils::replace $vide_cfg 'inputDataDir' "$wrk_dir/hod/$hod_hash"
utils::replace $vide_cfg 'workDir'      "$wrk_dir/hod/$hod_hash"
utils::replace $vide_cfg 'logDir'       "$logdir"
utils::replace $vide_cfg 'figDir'       "$figdir"
utils::replace $vide_cfg 'zmin'         "$zmin"
utils::replace $vide_cfg 'zmax'         "$zmax"
utils::replace $vide_cfg 'augment'      "$augment"

# prevent weird mutual interference effects between vide processes
cd $logdir

module load $VIDE_MODULES
conda activate $VIDE_CONDA_ENV

$VIDE_EXE $vide_cfg

conda deactivate
module rm $VIDE_MODULES
