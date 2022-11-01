from sys import argv
import os
from glob import glob
import subprocess

import numpy as np # necessary before pyglx
import galaxies.pyglx as pyglx # TODO does this work with relative paths?

# Script to be called from within HOD chains to create galaxies
# Command line arguments:
#   [1] working directory (e.g. /scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_0)
#   [2] hod hash (a hex string)
#   [3...] key=value pairs with keys from the below params dict

wrk_dir, hod_hash, argv_hod = argv[1], argv[2], argv[3:]

params = {
          'cat': pyglx.CatalogType,
          'secondary': pyglx.Secondary,
          'hod_log_Mmin': float,
          'hod_sigma_logM': float,
          'hod_log_M0': float,
          'hod_log_M1': float,
          'hod_alpha': float,
          'hod_transfP1': float,
          'hod_abias': float,
          'have_vbias': bool,
          'hod_transf_eta_cen': float,
          'hod_transf_eta_set': float
         }

def convert_argv(t, s) :
    if 'pyglx' in str(t) :
        return getattr(t, s)
    elif t == bool :
        return True if s=='True' else False
    else :
        return t(s)

kwargs = {k: convert_argv(params[k], v) for k, v in map(lambda s: s.split('='), argv_hod)}

# get the available snapshot times
# (try to do it a bit more precisely that 4 digits)
halo_finder_str = None
for a in argv_hod :
    if a.startswith('cat') :
        halo_finder_str = a.split('=')[-1]
        break
if halo_finder_str is None :
    halo_finder_str = 'rockstar'
halo_catalogs = sorted(glob(f'{wrk_dir}/{halo_finder_str}_*'), key=lambda s: float(s.split('_')[-1]))
times = []

# the pattern in the ascii file for the time
if halo_finder_str == 'rockstar' :
    pattern = '\#a\s=\s'
elif halo_finder_str == 'rfof' :
    pattern = 'Time.*\#HUMANE\s\[\s'

for ii, h in enumerate(halo_catalogs) :
    if halo_finder_str == 'rockstar' :
        header_file = f'{h}/out_{ii}_hosts.bf/Header/attr-v2'
    elif halo_finder_str == 'rfof' :
        header_file = f'{h}/Header/attr-v2'
    a = float(subprocess.run(f'grep -m1 -oP "{pattern}+\K\d.\d*" {header_file}',
                             shell=True, capture_output=True).stdout.strip())
    times.append(a)

# create our output directory if it doesn't exist
os.makedirs(f'{wrk_dir}/{hod_hash}', exist_ok=True)

# write the hod settings to file
with open(f'{wrk_dir}/{hod_hash}/hod.info', 'w') as f :
    f.write(f'hash={hod_hash}\n')
    for k, v in zip(params.keys(), argv_hod) :
        f.write(f'{k}={v}\n')

pyglx.get_galaxies(wrk_dir, times, f'{wrk_dir}/{hod_hash}/galaxies',
                   **kwargs)
