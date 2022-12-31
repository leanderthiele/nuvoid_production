from sys import argv
import os.path
from glob import glob
import re

import numpy as np

NBINS = int(argv[1])

Redges = np.linspace(30.0, 80.0, num=NBINS+1)
zedges = np.array([0.45, 0.53, 0.67])

results = {}

hod_hash = 'c2c93dbc97d64a7c20a043121f7d23d8'
root = '/scratch/gpfs/lthiele/nuvoid_production'
fid_dirs = sorted(glob(f'{root}/cosmo_fiducial_[0-9]*'))

for fid_dir in fid_dirs :
    
    print(fid_dir)
    
    seed_idx = int(re.search('(?<=cosmo_fiducial_)[0-9]*', fid_dir)[0])

    hists = np.empty((96, (len(Redges)-1)*(len(zedges)-1)), dtype=float)

    for ii in range(96) :

        voids_file = f'{fid_dir}/lightcones/{hod_hash}/voids_{ii}/sky_positions_central_{ii}.out'
        if not os.path.isfile(voids_file) :
            hists[ii] = float('nan')
            continue

        z, R = np.loadtxt(voids_file, usecols=(2,3), unpack=True)
        h = np.histogram2d(z, R, bins=[zedges, Redges])[0]
        hists[ii] = h.flatten().astype(float)

    if np.count_nonzero(np.isfinite(hists[:, 0])) > 70 :
        results[f'seed_{seed_idx}'] = hists

np.savez(f'fiducial_vsfs_{NBINS}bins.npz', Redges=Redges, zedges=zedges, **results)
