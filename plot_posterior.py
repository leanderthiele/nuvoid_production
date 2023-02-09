# TODO colors, legend, outfile name

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

DISCARD = 1000

# can pass multiple
chain_fname_bases = argv[1:]

# get the first one from corner
fig = None
a_leg = None
DIM = None
param_names = None

for chain_fname_base in chain_fname_bases :
    chain_fname_base_root, _ = os.path.splitext(chain_fname_base)

    chain_fname = f'{filebase}/{chain_fname_base}'
    with np.load(chain_fname) as f :
        chain = f['chain']
        logprob = f['log_prob']
        param_names_ = list(f['param_names'])
    if param_names is None :
        param_names = param_names_
    else :
        assert param_names == param_names_
    assert param_names[0] == 'Mnu'
    dim_ = len(param_names)
    if DIM is None :
        DIM = dim_
    else :
        assert DIM == dim_
    assert chain.shape[-1] == DIM
    chain = chain[DISCARD:, ...].reshape(-1, DIM)
    logprob = logprob[DISCARD:, ...].flatten()

    fig = corner.corner(chain, labels=param_names,
                        plot_datapoints=False, plot_density=False, no_fill_contours=True,
                        levels=1 - np.exp(-0.5 * np.array([1, 2])**2), # values in array are sigmas
                        fig=fig)

    # this is a bit hacky and depends on how corner implements stuff...
    a = fig.axes[0]
    p = a.patches[-1] # get the last one
    color = p.get_edgecolor()

    # this is where we put the legend
    if a_leg is None :
        a_leg = fig.axes[DIM-1]
    a_leg.hist(np.random.rand(2), label=chain_fname_base_root, color=color)

    if HAVE_PROFILE :
        xedges = p.xy[:, 0][::2]
        yvalues = p.xy[:, 1][1:-1:2]
        N = np.sum(yvalues)
        xcenters = 0.5*(xedges[1:] + xedges[:-1])
        mnu = chain[:, 0]
        bin_indices = np.digitize(mnu, xedges) - 1
        avg_logprob = np.empty(len(xedges)-1)
        for ii in range(len(xedges)-1) :
            avg_logprob[ii] = np.mean(logprob[bin_indices==ii])
        avg_logprob -= np.max(avg_logprob)
        avg_prob = np.exp(avg_logprob)
        avg_prob *= np.max(yvalues)
        a.plot(xcenters, avg_prob, linestyle='dotted', color=color)

# get the legend
a_leg.set_xlim(10,11)
a_leg.legend()

if HAVE_FS :
    # this is a bit hacky and depends on how corner implements stuff...
    a = fig.axes[0]
    p = a.patches[-1]
    xedges = p.xy[:, 0][::2]
    yvalues = p.xy[:, 1][1:-1:2]
    N = np.sum(yvalues)
    xcenters = 0.5*(xedges[1:] + xedges[:-1])
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


# for ii, name in enumerate(param_names) :
#     ax[ii, ii].set_title(name)

ident, _ = os.path.splitext(chain_fname_bases[0])
if len(chain_fname_bases) > 0 :
    ident = f'{ident}_and{len(chain_fname_bases)-1}'
fig.savefig(f'{filebase}/{ident}.pdf', bbox_inches='tight')
