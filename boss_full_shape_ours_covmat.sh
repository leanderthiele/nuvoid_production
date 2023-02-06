#!/bin/bash

module load anaconda3/2021.11
conda activate torch-env

wrkdir='<<<wrkdir>>>'

python -u /home/lthiele/montepython_public/montepython/MontePython.py info \
  --want-covmat \
  "$wrkdir"
