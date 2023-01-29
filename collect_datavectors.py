""" The ultimate script to assemble all data vectors
Command line arguments:
    [1] case (trials, fiducials_v%d, derivatives_v%d)
"""

import os.path
from sys import argv
import subprocess

import numpy as np

from datavector import VSF_ZEDGES, VSF_REDGES, VGPLK_R, VGPLK_ELL, VGPLK_K, PLK_ELL, PLK_K

codebase = '/home/lthiele/nuvoid_production'
database = '/scratch/gpfs/lthiele/nuvoid_production'

def get_datavec (path, lc) :
    """ return the data vector in path for lightcone lc,
        raises FileNotFoundError if any of the required files
        does not exist or is corrupted.
    """

    voids_fname = f'{path}/voids_{lc}/sky_positions_central_{lc}.out'
    if not os.path.isfile(voids_fname) :
        raise FileNotFoundError
    with open(voids_fname, 'r') as f :
        first_line = f.readline()
        if first_line[0] != '#' : # indicates corrupted file
            raise FileNotFoundError

    vgplk_fname = f'{path}/NEW_vgplk_{lc}.npz'
    if not os.path.isfile(vgplk_fname) :
        raise FileNotFoundError
    try :
        fvgplk = np.load(vgplk_fname)
    except ValueError : # indicates corrupted file
        raise FileNotFoundError
    if not np.allclose(VGPLK_K, fvgplk['k']) :
        raise FileNotFoundError

    plk_fname = f'{path}/NEW_plk_{lc}.npz'
    if not os.path.isfile(plk_fname) :
        raise FileNotFoundError
    try :
        fplk = np.load(plk_fname)
    except ValueError : # indicates corrupted file
        raise FileNotFoundError
    if not np.allclose(PLK_K, fplk['k']) :
        raise FileNotFoundError

    # compute VSF
    z, R = np.loadtxt(voids_fname, usecols=(2,3), unpack=True)
    out = np.histogram2d(z, R, bins=[VSF_ZEDGES, VSF_REDGES])[0].flatten().astype(float)

    # append VGPLK
    out = np.concatenate([out, *[fvgplk[f'p{ell}k_Rmin{R}'] for R in VGPLK_R for ell in VGPLK_ELL]])
    fvgplk.close()

    # append PLK
    out = np.concatenate([out, *[fplk[f'p{ell}k'] for ell in PLK_ELL]])
    fplk.close()

    return out


if __name__ == '__main__' :
    pass
