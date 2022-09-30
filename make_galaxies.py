# Command line arguments:
# [1] root directory
# [2] output redshift
# [3] output directory

from sys import argv

import numpy as np

import galaxies.pyglx as pyglx

ROOT = argv[1]
REDSHIFT = float(argv[2])
OUTDIR=argv[3]

TIME = 1.0/(1.0+REDSHIFT)

# HOD (gives approximately CMASS density at peak hopefully)
hod_ident = 'fidhod'
param_names = ['log_Mmin', 'sigma_logM', 'log_M0', 'log_M1', 'alpha', 'transf_eta_cen', 'transf_eta_sat', ]
theta0 = np.array([12.857,  0.33345, 13.55, 14.047,  0.41577,  1.2673, 0.23137])

hod = dict(zip(map(lambda s: 'hod_'+s, param_names), theta0))
result = pyglx.get_power([ROOT, ], REDSHIFT,
                         cat=pyglx.CatalogType.rockstar, rsd=pyglx.RSD.none, vgal_separate=True,
                         galaxies_bin_base='%s/galaxies_%s'%(OUTDIR, hod_ident),
                         galaxies_txt_base='%s/galaxies_%s'%(OUTDIR, hod_ident),
                         have_vbias=True, **hod)

np.savez('%s/power_%s_%.4f.npz'%(OUTDIR, hod_ident, TIME), k=result.k, Nk=result.Nk, Pk=result.Pk)
