# Small utility to prepare the jobs for a full-shape MCMC run
# Command line arguments:
#    [1] kmin
#    [2] kmax
#    [3] lmax
#    [4] AP (True or False)
#    [5] prior rescaling (float) -- if 0, use the no-As prior

kmin="$1"
kmax="$2"
lmax="$3"
AP="$4"
prior_rescale="$5"

codebase='/home/lthiele/nuvoid_production'
database='/scratch/gpfs/lthiele/nuvoid_production'

PARAMS_TEMPLATE="$codebase/boss_full_shape_ours.param"
TRIAL_TEMPLATE="$codebase/boss_full_shape_ours_trial.sbatch"
COVMAT_TEMPLATE="$codebase/boss_full_shape_ours_covmat.sh"
PRODUCTION_TEMPLATE="$codebase/boss_full_shape_ours_production.sbatch"

source utils.sh

ident="kmin${kmin}_kmax${kmax}_lmax${lmax}_AP${AP}_prior${prior_rescale}"

wrkdir_trial="$database/full_shape_trial_$ident"
wrkdir_production="$database/full_shape_production_$ident"
mkdir -p "$wrkdir_trial" "$wrkdir_production"

# set the parameter file
params_file="$wrkdir_trial/boss_full_shape_ours.param"
cp $PARAMS_TEMPLATE $params_file
utils::replace $params_file 'kmin' "$kmin"
utils::replace $params_file 'kmax' "$kmax"
utils::replace $params_file 'ellmax' "$lmax"
utils::replace $params_file 'have_AP' "$AP"
if [ "$prior_rescale" == "0" ]; then
  utils::replace $params_file 'cmb_prior' 'my_planck_prior_no_As'
  utils::replace $params_file 'prior_rescale' "1.0"
else
  utils::replace $params_file 'cmb_prior' 'my_planck_prior'
  utils::replace $params_file 'prior_rescale' "$prior_rescale"
fi

# the driver files
trial_file="$codebase/boss_full_shape_ours_trial_$ident.sbatch"
covmat_file="$codebase/boss_full_shape_ours_covmat_$ident.sh"
production_file="$codebase/boss_full_shape_ours_production_$ident.sbatch"

cp $TRIAL_TEMPLATE $trial_file
cp $COVMAT_TEMPLATE $covmat_file
cp $PRODUCTION_TEMPLATE $production_file

utils::replace $trial_file 'wrkdir' "$wrkdir_trial"
utils::replace $covmat_file 'wrkdir' "$wrkdir_trial"
utils::replace $production_file 'wrkdir_trial' "$wrkdir_trial"
utils::replace $production_file 'wrkdir_production' "$wrkdir_production"

utils::replace $trial_file 'params_file' "$params_file"
utils::replace $production_file 'params_file' "$params_file"
