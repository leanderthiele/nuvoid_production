# Call with VIDE galaxy file as first argument
# optional argument is output file name (by default just plk.npz)

from sys import argv

from mpi4py import MPI
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.io import fits
from astropy.stats import scott_bin_width
import nbodykit
from nbodykit.lab import *

# prevent nbodykit from doing funky stuff
nbodykit.CurrentMPIComm.set(MPI.COMM_SELF)


GAL_FILE = argv[1]
OUTFILE = argv[2] if len(argv)==3 else 'plk.npz'


h = 0.7
Om = 0.3
cosmo = cosmology.Cosmology(h=h).match(Omega0_m=Om)

P0 = 1e4

Nmesh = 512

# estimated using monte carlo
fsky = 0.16795440

def nz(z) :
    _, edges = scott_bin_width(z, return_bins=True)
    dig = np.searchsorted(edges, z, 'right')
    N = np.bincount(dig, minlength=len(edges)+1)[1:-1]
    R_hi = cosmo.comoving_distance(edges[1:])
    R_lo = cosmo.comoving_distance(edges[:-1])
    dV = 4*np.pi/3 * (R_hi**3 - R_lo**3) * fsky
    return InterpolatedUnivariateSpline(0.5*(edges[1:]+edges[:-1]), N/dV, ext='const')

ra_gals, dec_gals, z_gals = np.loadtxt(GAL_FILE, usecols=(3,4,5), unpack=True)
z_gals /= 299792.458

#with fits.open('/tigress/lthiele/boss_dr12/random0_DR12v5_CMASS_North.fits') as f :
#    ra_rand, dec_rand, z_rand = [f[1].data[s] for s in ['RA', 'DEC', 'Z']]
with np.load('/tigress/lthiele/boss_dr12/random_DR12v5_CMASS_North_downsampled6188060.npz') as f :
    ra_rand, dec_rand, z_rand = [f[s] for s in ['RA', 'DEC', 'Z']]

# remove the negative redshifts as they map to nan positions
select = z_gals > 0
ra_gals = ra_gals[select]
dec_gals = dec_gals[select]
z_gals = z_gals[select]
select = z_rand > 0
ra_rand = ra_rand[select]
dec_rand = dec_rand[select]
z_rand = z_rand[select]

ng_of_z = nz(z_gals)
nbar_gals = ng_of_z(z_gals)
nbar_rand = ng_of_z(z_rand)

pos_gals = transform.SkyToCartesian(ra_gals, dec_gals, z_gals, cosmo=cosmo)
pos_rand = transform.SkyToCartesian(ra_rand, dec_rand, z_rand, cosmo=cosmo)

w_gals = np.ones(len(z_gals))
w_rand = np.ones(len(z_rand))

fkp_gals = 1.0 / (1.0 + nbar_gals * P0)
fkp_rand = 1.0 / (1.0 + nbar_rand * P0)

cat_gals = ArrayCatalog({'Position': pos_gals, 'NZ': nbar_gals,
                         'WEIGHT': w_gals, 'WEIGHT_FKP': fkp_gals})
cat_rand = ArrayCatalog({'Position': pos_rand, 'NZ': nbar_rand,
                         'WEIGHT': w_rand, 'WEIGHT_FKP': fkp_rand})

cat_fkp = FKPCatalog(cat_gals, cat_rand)
mesh = cat_fkp.to_mesh(Nmesh=Nmesh, nbar='NZ', fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', window='tsc')

r = ConvolvedFFTPower(mesh, poles=[0,2,4], dk=0.01, kmax=0.401)

k = r.poles['k']
p0k = r.poles['power_0'].real - r.attrs['shotnoise']
p2k = r.poles['power_2'].real
p4k = r.poles['power_4'].real

np.savez(OUTFILE, k=k, p0k=p0k, p2k=p2k, p4k=p4k)
