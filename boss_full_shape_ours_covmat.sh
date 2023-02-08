#!/bin/bash

module load anaconda3/2021.11
module rm anaconda3/2021.11
module load anaconda3/2021.11
conda activate class-pt

wrkdir='<<<wrkdir>>>'

python -u /home/lthiele/montepython_public/montepython/MontePython.py info \
  --want-covmat \
  "$wrkdir"
