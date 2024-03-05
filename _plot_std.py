import os.path
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from _plot_style import *
from _plot_labels import *

CHAINSPATH_VARIED = '/tigress/lthiele/nuvoid_production/coverage_chains_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2'
DATFILE_VARIED = './std_varied.dat'

CHAINSPATH_FID = '/tigress/lthiele/nuvoid_production/'
DATFILE_FID = './std_fid.dat'

if not os.path.isfile(DATFILE_VARIED) :
    fnames = glob(f'{CHAINSPATH_VARIED}/*.npz')
    print(f'Found {len(fnames)} varied chains')
    true = []
    avg = []
    std = []
    for fname in fnames :
        with np.load(fname) as f :
            chain = f['chain'][200:, :, 0].flatten()
            t = f['real_params'][0]
            true.append(t)
            avg.append(np.mean(chain))
            std.append(np.std(chain))
    x = np.array([true, avg, std]).T
    np.savetxt(DATFILE_VARIED, x)

if not os.path.isfile(DATFILE_FID) :
    fnames = glob(f'{CHAINSPATH_FID}/lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_fid*_emceegpu.npz')
    print(f'Found {len(fnames)} fid chains before removing larger volumes')
    fnames = list(filter(lambda s: '000' not in s, fnames))
    print(f'Found {len(fnames)} fid chains after removing larger volumes')
    true = []
    avg = []
    std = []
    for fname in fnames :
        with np.load(fname) as f :
            chain = f['chain'][200:, :, 0].flatten()
            true.append(0.1)
            avg.append(np.mean(chain))
            std.append(np.std(chain))
    x = np.array([true, avg, std]).T
    np.savetxt(DATFILE_FID, x)

with np.load('/tigress/lthiele/nuvoid_production/lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_emceegpu.npz') as f :
    chain = f['chain'][200:, :, 0].flatten()
    real_std = np.std(chain)


tv, mv, sv = np.loadtxt(DATFILE_VARIED, unpack=True)
tf, mf, sf = np.loadtxt(DATFILE_FID, unpack=True)

fig, ax = plt.subplots(figsize=(5, 5))

mask = (tv>0.09) * (tv<0.11)
tv = tv[mask]
mv = mv[mask]
sv = sv[mask]

_, bins, _ = ax.hist(sv, bins=20, density=True, label='cosmo varied')
ax.hist(sf, bins=bins, alpha=0.5, density=True, label='fiducial')
ax.axvline(real_std, color='black', label='real data')


ax.set_yticks([])
ax.set_xlabel(f'{plot_labels["Mnu"]} standard deviation')
ax.legend()

savefig(fig, 'std')
