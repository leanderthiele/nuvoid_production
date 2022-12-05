from sys import argv

import numpy as np

datafile = argv[1]

zedges = np.linspace(0.42, 0.68, num=3)
Redges = np.linspace(50.0, 80.0, num=64)

zcenters = 0.5 * (zedges[1:] + zedges[:-1])
Rcenters = 0.5 * (Redges[1:] + Redges[:-1])

with np.load(datafile) as f :
    param_names = list(f['param_names'])
    cosmo_indices = f['cosmo_indices']
    params = f['params']
    radii = f['radii']
    redshifts = f['redshifts']

hists = np.stack([np.histogram2d(z_, r_, bins=(zedges, Redges))[0]
                 for z_, r_ in zip(radii, redshifts)])

np.savez('hists.npz',
         param_names=param_names,
         params=params,
         hists=hists,
         zedges=zedges,
         Redges=Redges,
         zcenters=zcenters,
         Rcenters=Rcenters)
