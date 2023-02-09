from sys import argv
import os.path
from glob import glob
import re

import numpy as np
from scipy.spatial import Delaunay

from matplotlib import pyplot as plt

import corner

from read_txt import read_txt
from lfi_load_posterior import load_posterior

# whether we overplot the full-shape posterior from the IAS group
HAVE_FS = True

# whether we also plot the profile likelihood
HAVE_PROFILE = False

filebase = '/tigress/lthiele/nuvoid_production'
fsroot = '/scratch/gpfs/lthiele/nuvoid_production'
fsruns = [
          'full_shape_production_kmin0.01_kmax0.15_lmax4',
          'full_shape_production_kmin0.01_kmax0.2_lmax4',
         ]

DISCARD = 1000

# can pass multiple
chain_fname_bases = list(map(os.path.basename, argv[1:]))

# get the first one from corner
fig = None
a_leg = None
DIM = None
param_names = None

color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def get_label (fname) :
    """ return a reasonable description of this posterior """
    match = re.search('.*v([0-9]*).*([a-f,0-9]{32}).*([a-f,0-9]{32}).*', fname)
    version = int(match[1])
    compression_hash = match[2]
    arch_hash = match[3]

    model_fname = f'{filebase}/lfi_model_v{version}_{compression_hash}_{arch_hash}.sbi'
    compression_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'

    model_settings, _ = load_posterior(model_fname, None, need_posterior=False)
    compression_settings = read_txt(compression_fname, 'cut_kwargs:', pyobj=True)

    label = f'$\\tt{{ {compression_hash[:4]}-{arch_hash[:4]} }}$'
    label += ', ' + '+'.join(map(lambda s: '$N_v$' if s=='vsf' \
                                      else '$P_\ell^{vg}$' if s=='vgplk' \
                                      else '$P_\ell^{gg}$' if s=='plk' \
                                      else s, \
                                 filter(lambda s: compression_settings[f'use_{s}'], ['vsf', 'vgplk', 'plk'])))
    if any(compression_settings[f'use_{s}'] for s in ['vgplk', 'plk', ]) :
        label += f', k={compression_settings["kmin"]:.2f}-{compression_settings["kmax"]:.2f}'
    label += f', {model_settings["model"][1]["num_blocks"]}x{model_settings["model"][1]["hidden_features"]}'

    return label


for chain_idx, chain_fname_base in enumerate(chain_fname_bases) :
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
                        color=color_cycle[chain_idx % len(color_cycle)],
                        fig=fig)

    # this is a bit hacky and depends on how corner implements stuff...
    a = fig.axes[0]
    p = a.patches[-1] # get the last one
    color = p.get_edgecolor()

    # this is where we put the legend
    if a_leg is None :
        a_leg = fig.axes[DIM-1]
    a_leg.hist(np.random.rand(2), label=get_label(chain_fname_base_root), color=color)

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
        match = re.search('.*(?<=kmin)([0-9,.]*).*(?<=kmax)([0-9,.]*).*', fsrun)
        kmin = float(match[1])
        kmax = float(match[2])
        l = a.plot(xcenters, h, linestyle='dashed')
        a_leg.plot(np.random.rand(2), linestyle=l[0].get_linestyle(), color=l[0].get_color(),
                   label=f'EFTofLSS, k={kmin:.2f}-{kmax:.2f}')

# get the legend
a_leg.set_xlim(10,11)
a_leg.legend()


# for ii, name in enumerate(param_names) :
#     ax[ii, ii].set_title(name)

ident, _ = os.path.splitext(chain_fname_bases[0])
if len(chain_fname_bases) > 0 :
    ident = f'{ident}_and{len(chain_fname_bases)-1}'
fig.savefig(f'{filebase}/{ident}.pdf', bbox_inches='tight')
