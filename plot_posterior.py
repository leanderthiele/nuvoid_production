from sys import argv

import numpy as np
from scipy.spatial import Delaunay

from matplotlib import pyplot as plt

import corner

HAVE_PRIOR = False

filebase = '/tigress/lthiele/nuvoid_production'

version = int(argv[1])
compression_hash = argv[2]
mode = argv[3]

if mode != 'mcmc' :
    HAVE_PRIOR = False

chain_fname = f'{filebase}/{mode}_chain_v{version}_{compression_hash}.npz'
with np.load(chain_fname) as f :
    chain = f['chain']
    param_names = list(f['param_names'])
DIM = len(param_names)
assert chain.shape[-1] == DIM
chain = chain.reshape(-1, DIM)

if HAVE_PRIOR :
    data_fname = f'{filebase}/avg_datavectors_trials.npz'
    with np.load(data_fname) as f :
        sim_idx = f['sim_idx']
        p = f['params']
        param_names_ = list(f['param_names'])
    param_indices = [param_names_.index(s) for s in param_names]
    p = p[:, param_indices]
    _, first_idx = np.unique(sim_idx, return_index=True)
    NCOSMO = 5
    p_LCDM = p[first_idx, :][:, :NCOSMO]
    del_tess_LCDM = Delaunay(p_LCDM)

    mu_cov_fname = f'{filebase}/mu_cov_plikHM_TTTEEE_lowl_lowE.dat'
    mu_LCDM = np.loadtxt(mu_cov_fname, max_rows=1)
    cov_LCDM = np.loadtxt(mu_cov_fname, skiprows=3)

    rng = np.random.default_rng(42)
    LCDM_prior_samples = rng.multivariate_normal(mu_LCDM, cov_LCDM, size=chain.shape[0]*(100000//43440))
    LCDM_prior_samples = LCDM_prior_samples[del_tess_LCDM.find_simplex(LCDM_prior_samples) != -1]
    fake_limits = [fct(chain[:, LCDM_prior_samples.shape[1]:], axis=0) for fct in [np.min, np.max]]
    fake_points = rng.uniform(*fake_limits, size=(LCDM_prior_samples.shape[0], len(fake_limits[0])))
    prior_samples = np.concatenate([LCDM_prior_samples, fake_points], axis=1)

fig, ax = plt.subplots(ncols=DIM, nrows=DIM, figsize=(30, 30))
if HAVE_PRIOR :
    corner.corner(prior_samples, plot_datapoints=False, fig=fig, color='grey')
corner.corner(chain, labels=param_names, plot_datapoints=False, fig=fig, color='blue')

for ii, name in enumerate(param_names) :
    ax[ii, ii].set_title(name)

fig.savefig(f'{filebase}/{mode}_chain_v{version}_{compression_hash}.pdf', bbox_inches='tight')
