from sys import argv

import numpy as np
from scipy.spatial import Delaunay

from matplotlib import pyplot as plt

import corner

DATA = argv[1]

CHAIN_FNAME = f'./{DATA}_mcmc_chain.npy'
DATA_FNAME = 'hists.npz'
PRIOR_FNAME = '/tigress/lthiele/mu_cov_plikHM_TTTEEE_lowl_lowE.dat'

with np.load(DATA_FNAME) as f :
    param_names = f['param_names']
    params = f['params'][:, :5]

tess = Delaunay(params)
DIM = len(param_names)
chain = np.load(CHAIN_FNAME).reshape(-1, DIM)
mu_prior = np.loadtxt(PRIOR_FNAME, max_rows=1)
cov_prior = np.loadtxt(PRIOR_FNAME, skiprows=3)
std_prior = np.sqrt(np.diag(cov_prior))

rng = np.random.default_rng(42)
prior_samples = rng.multivariate_normal(mu_prior, cov_prior, size=chain.shape[0]*(100000//43440))
s = tess.find_simplex(prior_samples)
prior_samples = prior_samples[s != -1]
fake_limits = [fct(chain[:, prior_samples.shape[1]:], axis=0) for fct in [np.min, np.max]]
fake_points = rng.uniform(*fake_limits, size=(prior_samples.shape[0], len(fake_limits[0])))
prior_samples = np.concatenate([prior_samples, fake_points], axis=1)

fig, ax = plt.subplots(ncols=DIM, nrows=DIM, figsize=(30, 30))

corner.corner(prior_samples, plot_datapoints=False, fig=fig, color='grey')
corner.corner(chain, labels=param_names, plot_datapoints=False, fig=fig, color='blue')

for ii, name in enumerate(param_names) :
    ax[ii, ii].set_title(name)

if False :
    for ii, (mu, sigma) in enumerate(zip(mu_prior, std_prior)) :
        lo, hi = ax[ii, ii].get_xlim()
        x = np.linspace(lo, hi)
        y = 1/np.sqrt(2*np.pi)/sigma * np.exp(-0.5 * (x-mu)**2/sigma**2)
        ylo, yhi = ax[ii, ii].get_ylim()
        y *= yhi / np.max(y)
        ax[ii, ii].plot(x, y)

fig.savefig(f'{DATA}_mcmc_posterior.pdf', bbox_inches='tight')
