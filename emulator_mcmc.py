from collections import OrderedDict
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.special import gammaln, xlogy
from scipy.spatial import Delaunay

import torch
import torch.nn as nn

import emcee

# whether we do not allow points outside the convex hull
# around our 5-D LCDM samples
# If we set this to False the LCDM posterior gets unreasonably wide
CONSTRAIN_CONVEX_HULL = True

MCMC_STEPS = 100000

# the cmass data
target_hist = np.array([58,62,51,54,47,44,41,42,42,25,30,19,18,27,14,9,9,5,7,4,6,1,2,2,1,0,0,1,2,0,0,0,57,56,67,48,45,53,41,45,49,37,39,50,49,36,28,17,17,33,15,13,14,7,5,12,6,6,2,4,0,1,2,0])

with np.load('/tigress/lthiele/emulator_data_RMIN30.0_RMAX80.0_NBINS32_ZEDGES0.53_untrimmed_dencut.npz') as f :
    param_names = list(f['param_names'])
    params = f['params']
    hists  = f['hists']
    values = f['values']

with np.load('vsf_mlp_norm.npz') as f :
    norm_avg = f['avg']
    norm_std = f['std']

# uniform priors for the HOD parameters + mnu
NCOSMO = 5
hod_theta_min = np.min(params[:, NCOSMO:], axis=0)
hod_theta_max = np.max(params[:, NCOSMO:], axis=0)
delta = hod_theta_max - hod_theta_min
eps = 0.01
hod_theta_min += eps * delta
hod_theta_max -= eps * delta

# the Gaussian prior on 5-parameter LCDM
mu_cov_fname = '/tigress/lthiele/mu_cov_plikHM_TTTEEE_lowl_lowE.dat'
mu_LCDM = np.loadtxt(mu_cov_fname, max_rows=1)
cov_LCDM = np.loadtxt(mu_cov_fname, skiprows=3)

# more complicated prior for the cosmology
if CONSTRAIN_CONVEX_HULL :
    del_tess = Delaunay(params[:, :NCOSMO])

class MLPLayer(nn.Sequential) :
    def __init__(self, Nin, Nout, activation=nn.LeakyReLU) :
        super().__init__(OrderedDict([('linear', nn.Linear(Nin, Nout, bias=True)),
                                      ('activation', activation()),
                                     ]))

class MLP(nn.Sequential) :
    def __init__(self, Nin, Nout, Nlayers=8, Nhidden=512) :
        # output is manifestly positive so we use ReLU in the final layer
        super().__init__(*[MLPLayer(Nin if ii==0 else Nhidden,
                                    Nout if ii==Nlayers else Nhidden,
                                    activation=(nn.LeakyReLU if ii!=Nlayers else nn.ReLU))
                           for ii in range(Nlayers+1)])

model = MLP(params.shape[1], hists.shape[1])
model.load_state_dict(torch.load('vsf_mlp.pt', map_location='cpu'))


def logprior(theta) :
    
    # hod+mnu part
    if not np.all((hod_theta_min<=theta[NCOSMO:])*(theta[NCOSMO:]<=hod_theta_max)) :
        return -np.inf

    # make sure emulator is valid
    if CONSTRAIN_CONVEX_HULL and del_tess.find_simplex(theta[:NCOSMO]).item() == -1 :
        return -np.inf
    
    # LCDM part
    d = theta[:NCOSMO] - mu_LCDM
    lp_lcdm = -0.5 * np.einsum('i,ij,j->', d, cov_LCDM, d)
    return lp_lcdm


def loglike(theta) :
    theta = (theta - norm_avg) / norm_std
    mu = model(torch.from_numpy(theta).to(dtype=torch.float32)).detach().cpu().numpy() + 1e-8
    return np.sum(xlogy(target_hist, mu) - mu - gammaln(1.0+target_hist))


def logprob(theta) :
    lp = logprior(theta)
    if not np.isfinite(lp) :
        return -np.inf
    ll = loglike(theta)
    return lp + ll


if __name__ == '__main__' :

    NWALKERS = 128
    NDIM = params.shape[1]

    s = np.argsort(values)[::-1]
    theta_init = params[s[:NWALKERS]]

    print(f'Running on {cpu_count()} CPUs')
    with Pool() as pool :
        sampler = emcee.EnsembleSampler(NWALKERS, NDIM, logprob, pool=pool)

        sampler.run_mcmc(theta_init, MCMC_STEPS, progress=True)

        chain = sampler.get_chain(thin=30, discard=MCMC_STEPS//5)
        np.save('vsf_mcmc_chain.npy', chain)

        acceptance_rates = sampler.acceptance_fraction
        print(f'acceptance={acceptance_rates}')

        autocorr_times = sampler.get_autocorr_time()
        print(f'autocorr={autocorr_times}')
