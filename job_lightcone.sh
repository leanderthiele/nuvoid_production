#!/bin/bash

#SBATCH -n 48
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 00:10:00

set -e -x -o pipefail

# =========== CONFIGURATION OPTIONS ==========
# these are really the only things we want to change from sbatch to sbatch script

# use this to uniquely identify runs, replace by array or similar later
export ID="test1"

# size of the simulation
export NC=2800

# random FastPM seed
export SEED=137

# =========== GENERAL QUIJOTE =============
# exports BOX_SIZE, Z_OUT, BASE
source cmass.sh

# =========== COSMOLOGY =============
# exports all COSMO_* variables
source mnu200meV_cosmo.sh

# =========== RUN CODES ===========
# all snapshots:
# 0.5965 0.6074 0.6152 0.6216 0.6270 0.6320 0.6367 0.6409 0.6449 0.6488 0.6526 0.6563 0.6601 0.6638 0.6676 0.6715 0.6757 0.6803 0.6856 0.6930

# the different sampling schemes
samples=(
         # [0] 20 : reference
         "0.5965 0.6074 0.6152 0.6216 0.6270 0.6320 0.6367 0.6409 0.6449 0.6488 0.6526 0.6563 0.6601 0.6638 0.6676 0.6715 0.6757 0.6803 0.6856 0.6930"
         # [1] 11 : worse than the 10
         "0.5965        0.6152        0.6270        0.6367        0.6449        0.6526        0.6601        0.6676        0.6757        0.6856 0.6930"
         # [2] 8 : worse than the other 8
         "0.5965               0.6216               0.6367               0.6488               0.6601               0.6715        0.6803        0.6930"
         # [3] 10 : better than 11, about same as better 8
         "       0.6074        0.6216        0.6320        0.6409        0.6488        0.6563        0.6638        0.6715        0.6803        0.6930"
         # [4] 6 : bad
         "0.5965                      0.6270                      0.6449                      0.6601                      0.6757               0.6930"
         # [5] 8 : better than the other 8
         "       0.6074                      0.6320        0.6409        0.6488        0.6563        0.6638                      0.6803        0.6930"
         # [6] 5 : bad
         "       0.6074                             0.6367                             0.6563                             0.6757               0.6930"
         # [7] 8 : ok, similar to [5]
         "       0.6074               0.6270        0.6367        0.6449        0.6526        0.6601               0.6715               0.6856       "
         # [8] 8 : bad
         "       0.6074                      0.6320        0.6409        0.6488        0.6563        0.6638               0.6757               0.6930"
         # [9] 10 : ok, similar to [5]
         "       0.6074                      0.6320        0.6409        0.6488 0.6526 0.6563        0.6638        0.6715        0.6803        0.6930"
         # [10] 11 : ok, similar to [5]
         "       0.6074        0.6216        0.6320        0.6409        0.6488 0.6526 0.6563        0.6638        0.6715        0.6803        0.6930"
         # [11] 9 : bad
         "       0.6074                      0.6320        0.6409        0.6488 0.6526 0.6563        0.6638               0.6757               0.6930"
         # [12] 9 : ?
         "       0.6074        0.6216        0.6320               0.6449               0.6563               0.6676        0.6757        0.6856 0.6930"
         # [13] 9 : ?
         "       0.6074        0.6216        0.6320               0.6449               0.6563               0.6676               0.6803 0.6856 0.6930"
         # [14] 9 : ?
         "       0.6074        0.6216        0.6320               0.6449               0.6563               0.6676               0.6803 0.6856       "
         # [15] 10 : ?
         "       0.6074        0.6216        0.6320               0.6449               0.6563               0.6676        0.6757 0.6803 0.6856       "
         # [16] 9 : ?
         "       0.6074                      0.6320               0.6449               0.6563               0.6676        0.6757 0.6803 0.6856       "
         # [17] 8 : ?
         "       0.6074                      0.6320               0.6449               0.6563               0.6676        0.6757        0.6856 0.6930"
         # [18] 9 : ?
         "0.5965        0.6152               0.6320               0.6449               0.6563               0.6676        0.6757        0.6856 0.6930"
        )

# BIMODAL behaviour at low z
# MODE 1 -- too high at low radii, too low at large radii
# But these are generally better (lower chisquared)
         "       0.6074                      0.6320        0.6409        0.6488 0.6526 0.6563        0.6638        0.6715        0.6803        0.6930"
         "       0.6074        0.6216        0.6320        0.6409        0.6488 0.6526 0.6563        0.6638        0.6715        0.6803        0.6930"
         "       0.6074        0.6216        0.6320               0.6449               0.6563               0.6676               0.6803 0.6856 0.6930"
         "       0.6074        0.6216        0.6320               0.6449               0.6563               0.6676               0.6803 0.6856       "
# MODE 2 -- too low at low radii, too high at large radii
         "       0.6074                      0.6320        0.6409        0.6488        0.6563        0.6638               0.6757               0.6930"
         "       0.6074                      0.6320        0.6409        0.6488 0.6526 0.6563        0.6638        0.6715        0.6803        0.6930"
         "       0.6074        0.6216        0.6320               0.6449               0.6563               0.6676        0.6757        0.6856 0.6930"


export REMAP_CASE=0
export CORRECT=0
export VETO=1

ii=0

for time_samples in "${samples[@]}"; do

  if [ $ii -ge 12 ]; then
    echo "$time_samples" >> "/scratch/gpfs/lthiele/nuvoid_production/test1/galaxies/time_samples_$ii.info"
    srun -n 48 -W 0 bash lightcone.sh $ii "$time_samples"
  fi

  ii=$((ii+1))
done
