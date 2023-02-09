from sys import argv
import os.path
from glob import glob

import numpy as np
from scipy.spatial import Delaunay

from matplotlib import pyplot as plt

import corner

# whether we overplot the full-shape posterior from the IAS group
HAVE_FS = True

# whether we also plot the profile likelihood
HAVE_PROFILE = True

filebase = '/tigress/lthiele/nuvoid_production'
fsroot = '/scratch/gpfs/lthiele/nuvoid_production'
fsruns = [
          'full_shape_production_kmin0.01_kmax0.15_lmax4',
          'full_shape_production_kmin0.01_kmax0.2_lmax4',
         ]

chain_fname_base = argv[1]
chain_fname_base_root, _ = os.path.splitext(chain_fname_base)

chain_fname = f'{filebase}/{chain_fname_base}'
with np.load(chain_fname) as f :
    chain = f['chain']
    logprob = f['log_prob']
    param_names = list(f['param_names'])
assert param_names[0] == 'Mnu'
DIM = len(param_names)
assert chain.shape[-1] == DIM
chain = chain.reshape(-1, DIM)
logprob = logprob.flatten()

fig = corner.corner(chain, labels=param_names, plot_datapoints=False, color='black')

if HAVE_FS or HAVE_PROFILE :
    # this is a bit hacky and depends on how corner implements stuff...
    a = fig.axes[0]
    p = a.patches[-1]
    xedges = p.xy[:, 0][::2]
    yvalues = p.xy[:, 1][1:-1:2]
    N = np.sum(yvalues)
    xcenters = 0.5*(xedges[1:] + xedges[:-1])
    if HAVE_FS :
        for fsrun in fsruns :
            fsbase = f'{fsroot}/{fsrun}'

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
                fs_Mnu_ = np.repeat(fs_Mnu_, repeats.astype(int))[200:] # discard burn-in
                fs_Mnu = np.concatenate((fs_Mnu, fs_Mnu_))
            h, e = np.histogram(fs_Mnu, bins=xedges)
            h = h.astype(float) * N / np.sum(h)
            a.plot(xcenters, h, linestyle='dashed')
    if HAVE_PROFILE :
        mnu = chain[:, 0]
        bin_indices = np.digitize(mnu, xedges) - 1
        avg_logprob = np.empty(len(xedges)-1)
        for ii in range(len(xedges)-1) :
            avg_logprob[ii] = np.mean(logprob[bin_indices==ii])
        avg_logprob -= np.max(avg_logprob)
        avg_prob = np.exp(avg_logprob)
        avg_prob *= np.max(yvalues)
        a.plot(xcenters, avg_prob, linestyle='dotted', color='black')


# for ii, name in enumerate(param_names) :
#     ax[ii, ii].set_title(name)

fig.savefig(f'{filebase}/{chain_fname_base_root}.pdf', bbox_inches='tight')
