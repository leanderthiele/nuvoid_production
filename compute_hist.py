from sys import argv

import numpy as np

datafile = argv[1]

zedges = np.linspace(0.42, 0.68, num=3)
Redges = np.linspace(40.0, 80.0, num=80)

zcenters = 0.5 * (zedges[1:] + zedges[:-1])
Rcenters = 0.5 * (Redges[1:] + Redges[:-1])

with np.load(datafile) as f :
    param_names = list(f['param_names'])
    cosmo_indices = f['cosmo_indices']
    params = f['params']
    radii = f['radii']
    redshifts = f['redshifts']

def get_hist(z, r) :
    return np.histogram2d(z, r, bins=(zedges, Redges))[0]

hists = np.stack([get_hist(z_, r_) for z_, r_ in zip(redshifts, radii)])

r_cmass, z_cmass = np.loadtxt('/tigress/lthiele/boss_dr12/voids/sample_test/untrimmed_dencut_centers_central_test.out',
                              usecols=(4,5), unpack=True)

hist_cmass = get_hist(z_cmass, r_cmass)

np.savez('hists.npz',
         param_names=param_names,
         params=params,
         hists=hists,
         hist_cmass=hist_cmass,
         zedges=zedges,
         Redges=Redges,
         zcenters=zcenters,
         Rcenters=Rcenters)
