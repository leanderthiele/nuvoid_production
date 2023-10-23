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
          'full_shape_production_kmin0.01_kmax0.15_lmax0_APTrue',
          'full_shape_production_kmin0.01_kmax0.2_lmax4',
#          'full_shape_production_kmin0.01_kmax0.15_lmax2_APFalse',
          'full_shape_production_kmin0.01_kmax0.2_lmax0_APTrue',
#          'full_shape_production_kmin0.01_kmax0.1_lmax2',
         ]

DISCARD = 1000

# chains can have different sizes due to different numbers of CPUs used
# it is most convenient to just subsample them to a common size
USE_SAMPLES = 16 * 10000

# can pass multiple
if argv[-1] == 'nofs' :
    HAVE_FS = False
    args = argv[1:-1]
else :
    args = argv[1:]
set_colors = []
for ii, a in enumerate(args) :
    match = re.search('(.*)\(([a-z]*)\)', a)
    if match is not None :
        set_colors.append(match[2])
        args[ii] = match[1]
    else :
        set_colors.append(None)
chain_fname_bases = list(map(os.path.basename, args))

# get the first one from corner
fig = None
figdummy = None
a_leg = None
DIM = None
param_names = None
idents = []

color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
linestyle_cycle = ['-', '--', ':', '-.', ]
rng = np.random.default_rng(42)

def get_label (fname) :
    """ return a reasonable description of this posterior """
    match = re.search('.*v([0-9]*).*([a-f,0-9]{32}).*([a-f,0-9]{32}).*', fname)
    version = int(match[1])
    compression_hash = match[2]
    arch_hash = match[3]

    if 'fid' in fname :
        match = re.search('.*fid([0-9]*).*', fname)
        fid_idx = int(match[1])
    else :
        fid_idx = None

    ident = f'{compression_hash[:4]}-{arch_hash[:4]}'

    model_fname = f'{filebase}/lfi_model_v{version}_{compression_hash}_{arch_hash}.sbi'
    compression_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'

    model_settings, _ = load_posterior(model_fname, None, need_posterior=False)
    compression_settings = read_txt(compression_fname, 'cut_kwargs:', pyobj=True)

    label = f'$\\tt{{ {ident} }}$'
    label += ', ' + '+'.join(map(lambda s: '$N_v$' if s=='vsf' \
                                        else f'$P_{{{",".join(map(str, sorted(compression_settings["vgplk_ell"])))}}}^{{vg}}$' if s=='vgplk' \
                                      else f'$P_{{{",".join(map(str, sorted(compression_settings["plk_ell"])))}}}^{{gg}}$' if s=='plk' \
                                      else s, \
                                 filter(lambda s: compression_settings[f'use_{s}'], ['vsf', 'vgplk', 'plk'])))
    if any(compression_settings[f'use_{s}'] for s in ['vgplk', 'plk', ]) :
        label += f', k={compression_settings["kmin"]:.2f}-{compression_settings["kmax"]:.2f}'
    label += f', {model_settings["model"][1]["num_blocks"]}x{model_settings["model"][1]["hidden_features"]}'
    if fid_idx is not None :
        label += f', fiducial {fid_idx}'

    return label, ident


color_idx = 0
for chain_fname_base, set_color in zip(chain_fname_bases, set_colors) :
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

    rnd_indices = rng.choice(len(chain), size=USE_SAMPLES, replace=False)
    chain = chain[rnd_indices]
    logprob = logprob[rnd_indices]

    if set_color is not None :
        color = set_color
    else :
        color = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1
    fig = corner.corner(chain, labels=param_names,
                        plot_datapoints=False, plot_density=False, no_fill_contours=True,
                        levels=1 - np.exp(-0.5 * np.array([2])**2), # values in array are sigmas
                        color=color,
                        smooth1d=0.01, hist_bin_factor=2,
                        fig=fig)

    # to get the patches from which to construct the histograms
    figdummy = corner.corner(chain, labels=param_names,
                             plot_datapoints=False, plot_density=False, no_fill_contours=True,
                             levels=1 - np.exp(-0.5 * np.array([1, 2])**2), # values in array are sigmas
                             color=color,
                             fig=figdummy)

    # this is a bit hacky and depends on how corner implements stuff...
    p = figdummy.axes[0].patches[-1] # get the last one
    color = p.get_edgecolor()

    # this is where we put the legend
    if a_leg is None :
        a_leg = fig.axes[1]
    label, ident = get_label(chain_fname_base_root)
    idents.append(ident)
    a_leg.hist(np.random.rand(2), label=label, color=color)

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
        avg_prob *= np.sum(yvalues)/np.sum(avg_prob)
        fig.axes[0].plot(xcenters, avg_prob, linestyle='dotted', color=color)

if HAVE_FS :
    # this is a bit hacky and depends on how corner implements stuff...
    p = figdummy.axes[0].patches[-1]
    xedges = p.xy[:, 0][::2]
    yvalues = p.xy[:, 1][1:-1:2]
    N = np.sum(yvalues)
    xcenters = 0.5*(xedges[1:] + xedges[:-1])
    for fsrun_idx, fsrun in enumerate(fsruns) :
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
        match = re.search('.*(?<=kmin)([0-9,.]*).*(?<=kmax)([0-9,.]*).*(?<=lmax)([0-9]).*', fsrun)
        kmin = float(match[1])
        kmax = float(match[2])
        lmax = int(match[3])
        label = f'EFTofLSS $P^{{gg}}_{{{",".join(map(str, range(0, lmax+1, 2)))}}}$, $k$={kmin:.2f}-{kmax:.2f}'
        if 'APFalse' in fsrun :
            label += ', no AP'
        l = fig.axes[0].plot(xcenters, h, linestyle=linestyle_cycle[fsrun_idx % len(linestyle_cycle)],
                             color='grey')
        a_leg.plot(np.random.rand(2), linestyle=l[0].get_linestyle(), color=l[0].get_color(),
                   label=label)

# get the legend
a_leg.set_xlim(10,11)
a_leg.legend(frameon=False, fontsize='x-small', loc='upper left')


# for ii, name in enumerate(param_names) :
#     ax[ii, ii].set_title(name)

fig.savefig(f'{filebase}/posteriors_{"_".join(idents)}.pdf', bbox_inches='tight')
