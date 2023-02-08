from sys import argv
import os.path
from glob import glob

import numpy as np
from scipy.spatial import Delaunay

from matplotlib import pyplot as plt

import corner

filebase = '/tigress/lthiele/nuvoid_production'
fsroot = '/scratch/gpfs/lthiele/nuvoid_production'
fsrun = 'full_shape_production_kmin0.03_kmax0.15_lmax2'

fsbase = f'{fsroot}/{fsrun}'

chain_fname_base = argv[1]
chain_fname_base_root, _ = os.path.splitext(chain_fname_base)

chain_fname = f'{filebase}/{chain_fname_base}'
with np.load(chain_fname) as f :
    chain = f['chain']
    param_names = list(f['param_names'])
assert param_names[0] == 'Mnu'
DIM = len(param_names)
assert chain.shape[-1] == DIM
chain = chain.reshape(-1, DIM)

fs_param_names_files = glob(f'{fsbase}/*.paramnames')
assert len(fs_param_names_files) == 1
Mnu_idx = None
with open(fs_param_names_files[0], 'r') as f :
    for ii, line in enumerate(f) :
        if line.startswith('M_tot') :
            Mnu_idx = ii
            break
assert Mnu_idx is not None
fs_Mnu = np.empty(0)

fs_txt_files = glob(f'{fsbase}/*.txt')
for fs_txt_file in fs_txt_files :
    repeats, fs_Mnu_ = np.loadtxt(fs_txt_file, usecols=(0, 2+Mnu_idx), unpack=True)
    fs_Mnu_ = np.repeat(fs_Mnu_, repeats)[200:] # discard burn-in
    fs_Mnu = np.concatenate((fs_Mnu, fs_Mnu_))

fig = corner.corner(chain, labels=param_names, plot_datapoints=False, color='blue')

# this is a bit hacky and depends on how corner implements stuff...
a = fig.axes[0]
p = a.patches[-1]
xedges = p.xy[:, 0][::2]
yvalues = p.xy[:, 1][1:-1:2]
N = np.sum(yvalues)
h, e = np.histogram(fs_Mnu, bins=xedges)
h = h.astype(float) * N / np.sum(h)
c = 0.5*(e[1:] + e[:-1])
a.plot(c, h, linestyle='dashed', color='red')

# for ii, name in enumerate(param_names) :
#     ax[ii, ii].set_title(name)

fig.savefig(f'{filebase}/{chain_fname_base_root}.pdf', bbox_inches='tight')
