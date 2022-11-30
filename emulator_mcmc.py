from collections import OrderedDict

import numpy as np
from scipy.special import gammaln, xlogy

import torch
import torch.nn as nn

import emcee

# the cmass data
target_hist = np.array([58,62,51,54,47,44,41,42,42,25,30,19,18,27,14,9,9,5,7,4,6,1,2,2,1,0,0,1,2,0,0,0,57,56,67,48,45,53,41,45,49,37,39,50,49,36,28,17,17,33,15,13,14,7,5,12,6,6,2,4,0,1,2,0])

with np.load('/tigress/lthiele/emulator_data_RMIN30.0_RMAX80.0_NBINS32_ZEDGES0.53_untrimmed_dencut.npz') as f :
    param_names = list(f['param_names'])
    params = f['params']
    hists  = f['hists']

theta_min = np.min(params, axis=0)
theta_max = np.max(params, axis=0)
delta = theta_max - theta_min
eps = 0.05
theta_min += eps * delta
theta_max -= eps * delta

class MLPLayer(nn.Sequential) :
    def __init__(self, Nin, Nout, activation=nn.LeakyReLU) :
        super().__init__(OrderedDict([('linear', nn.Linear(Nin, Nout, bias=True)),
                                      ('activation', activation()),
                                     ]))

class MLP(nn.Sequential) :
    def __init__(self, Nin, Nout, Nlayers=4, Nhidden=512) :
        # output is manifestly positive so we use ReLU in the final layer
        super().__init__(*[MLPLayer(Nin if ii==0 else Nhidden,
                                    Nout if ii==Nlayers else Nhidden,
                                    activation=(nn.LeakyReLU if ii!=Nlayers else nn.ReLU))
                           for ii in range(Nlayers+1)])

model = MLP(params.shape[1], hists.shape[1])
model.load_state_dict(torch.load('vsf_mlp.pt', map_location='cpu'))

def logprior(theta) :
    if np.all((theta_min<=theta)*(theta<=theta_max)) :
        return 0.0
    return -np.inf

def loglike(theta) :
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

    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, logprob)

    rng = np.random.default_rng(42)
    theta_init = rng.uniform(theta_min, theta_max, size=(NWALKERS, NDIM))
    sampler.run_mcmc(theta_init, 100000, progress=True)

    chain = sampler.get_chain()
    np.save('vsf_mcmc_chain.npy', chain)

    autocorr_times = sampler.get_autocorr_time()
    acceptance_rates = sampler.acceptance_fraction

    print(f'autocorr={autocorr_times}\n, acceptance={acceptance_rates}')
