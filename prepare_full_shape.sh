# Small utility to prepare the jobs for a full-shape MCMC run
# Command line arguments:
#    [1] kmin
#    [2] kmax
#    [3] lmax

kmin="$1"
kmax="$2"
lmax="$3"

codebase='/home/lthiele/nuvoid_production'
database='/scratch/gpfs/lthiele/nuvoid_production'

PARAMS_TEMPLATE="$codebase/boss_full_shape_ours.param"
TRIAL_TEMPLATE="$codebase/boss_full_shape_ours_trial.sbatch"
COVMAT_TEMPLATE="$codebase/boss_full_shape_ours_covmat.sh"
PRODUCTION_TEMPLATE="$codebase/boss_full_shape_ours_production.sbatch"

source utils.sh

wrkdir_trial="$database/full_shape_trial_kmin${kmin}_kmax${kmax}_lmax${lmax}"
wrkdir_production="$database/full_shape_production_kmin${kmin}_kmax${kmax}_lmax${lmax}"
mkdir -p "$wrkdir_trial" "$wrkdir_production"

# set the parameter file
params_file="$wrkdir_trial/boss_full_shape_ours.param"
cp $PARAMS_TEMPLATE $params_file
utils::replace $params_file 'kmin' "$kmin"
utils::replace $params_file 'kmax' "$kmax"
utils::replace $params_file 'ellmax' "$lmax"

# the driver files
trial_file="$codebase/boss_full_shape_ours_trial_kmin${kmin}_kmax${kmax}_lmax${lmax}.sbatch"
covmat_file="$codebase/boss_full_shape_ours_covmat_kmin${kmin}_kmax${kmax}_lmax${lmax}.sh"
production_file="$codebase/boss_full_shape_ours_production_kmin${kmin}_kmax${kmax}_lmax${lmax}.sbatch"

cp $TRIAL_TEMPLATE $trial_file
cp $COVMAT_TEMPLATE $covmat_file
cp $PRODUCTION_TEMPLATE $production_file

utils::replace $trial_file 'wrkdir' "$wrkdir_trial"
utils::replace $covmat_file 'wrkdir' "$wrkdir_trial"
utils::replace $production_file 'wrkdir_trial' "$wrkdir_trial"
utils::replace $production_file 'wrkdir_production' "$wrkdir_production"

utils::replace $trial_file 'params_file' "$params_file"
utils::replace $production_file 'params_file' "$params_file"
