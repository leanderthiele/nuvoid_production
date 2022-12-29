import os.path
import re

import numpy as np

from glob import glob

hod_hash = 'c2c93dbc97d64a7c20a043121f7d23d8'
root = '/scratch/gpfs/lthiele/nuvoid_production'
fid_dirs = glob(f'{root}/cosmo_fiducial_[0-9]*')

k = None
Rmin = None

results = {}

for fid_dir in fid_dirs :
    
    print(fid_dir)

    seed_idx = int(re.search('(?<=cosmo_fiducial_)[0-9]*', fid_dir)[0])

    vgplk = np.empty((96, 3, 2, 40), dtype=float)

    for ii in range(96) :
        
        vgplk_file = f'{fid_dir}/lightcones/{hod_hash}/NEW_vgplk_{ii}.npz'
        if not os.path.isfile(vgplk_file) :
            vgplk[ii] = float('nan')
            continue

        with np.load(vgplk_file) as f :
            k_ = f['k']
            if k is None :
                k = k_
                assert len(k) == vgplk.shape[-1]
            else :
                assert np.allclose(k, k_)
            Rmin_ = np.array(sorted(list(map(lambda s: int(s.split('Rmin')[1]),
                                         filter(lambda s: 'p0k' in s, list(f.keys()))))))
            if Rmin is None :
                Rmin = Rmin_
                assert len(Rmin) == vgplk.shape[1]
            else :
                assert np.allclose(Rmin, Rmin_)
            for jj, Rmin_ in enumerate(Rmin) :
                for kk, ell in enumerate([0, 2]) :
                    vgplk[ii, jj, kk, :] = f[f'p{ell}k_Rmin{Rmin_}']

    if np.count_nonzero(np.isfinite(vgplk[:, 0, 0, 0])) > 70 :
        results[f'seed_{seed_idx}'] = vgplk

np.savez('fiducial_vgplk.npz', k=k, Rmin=Rmin, **results)
