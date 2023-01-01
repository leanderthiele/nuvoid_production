from collections import OrderedDict
from multiprocessing import Pool, cpu_count

import numpy as np

import torch
import torch.nn as nn

import emcee

# needs to be consistent!
RMIN = 40
USE_QUADRUPOLE = False

# whether we do not allow points outside the convex hull
# around our 5-D LCDM samples
# If we set this to False the LCDM posterior gets unreasonably wide
CONSTRAIN_CONVEX_HULL = True

MCMC_STEPS = 10000

with np.load('/tigress/lthiele/cov_vgplk.npz') as f :
    cov = f['cov'] # shape [Rmin, ellxk]
    assert np.allclose(Rmin, f['Rmin'])
    k_indices = f['k_indices']
cov = cov[rmin_idx, ...]
# remove the quadrupole if requested
if not USE_QUADRUPOLE :
    cov = cov[:len(k_indices), :len(k_indices)]
covinv = np.linalg.inv(cov)

with np.load('vgplk_norm.npz') as f :
    norm_avg = f['avg']
    norm_std = f['std']

with np.load('/tigress/lthiele/collected_vgplk.npz') as f :
    param_names = list(f['param_names'])
    params = f['params']

with np.load('/tigress/lthiele/cmass_vgplk.npz') as f :
    target = f[f'p0k_Rmin{RMIN}'][k_indices]
    if USE_QUADRUPOLE :
        target = np.concatenate([target, f[f'p2k_Rmin{RMIN}'][k_indices]])

# uniform priors for the HOD parameters + mnu
NCOSMO = 5
hod_theta_min = np.min(params[:, NCOSMO:], axis=0)
hod_theta_max = np.max(params[:, NCOSMO:], axis=0)
delta = hod_theta_max - hod_theta_min
eps = 0.01
hod_theta_min += eps * delta
hod_theta_max -= eps * delta

# more complicated prior for the cosmology
if CONSTRAIN_CONVEX_HULL :
    del_tess = Delaunay(params[:, :NCOSMO])

# the Gaussian prior on 5-parameter LCDM
mu_cov_fname = '/tigress/lthiele/mu_cov_plikHM_TTTEEE_lowl_lowE.dat'
mu_LCDM = np.loadtxt(mu_cov_fname, max_rows=1)
cov_LCDM = np.loadtxt(mu_cov_fname, skiprows=3)
covinv_LCDM = np.linalg.inv(cov_LCDM)

class MLPLayer(nn.Sequential) :
    def __init__(self, Nin, Nout, activation=nn.LeakyReLU) :
        super().__init__(OrderedDict([('linear', nn.Linear(Nin, Nout, bias=True)),
                                      ('activation', activation()),
                                     ]))

class MLP(nn.Sequential) :
    def __init__(self, Nin, Nout, Nlayers=4, Nhidden=512, out_positive=True) :
        # output is manifestly positive so we use ReLU in the final layer
        self.Nin = Nin
        self.Nout = Nout
        super().__init__(*[MLPLayer(Nin if ii==0 else Nhidden,
                                    Nout if ii==Nlayers else Nhidden,
                                    activation=(nn.LeakyReLU if (ii!=Nlayers or not out_positive) else nn.ReLU))
                           for ii in range(Nlayers+1)])

model = MLP(params.shape[-1], (1+USE_QUADRUPOLE)*len(k_indices))
model.load_state_dict(torch.load('vgplk_mlp.pt', map_location='cpu')

def logprior(theta) :
    
    # hod+mnu part
    if not np.all((hod_theta_min<=theta[NCOSMO:])*(theta[NCOSMO:]<=hod_theta_max)) :
        return -np.inf

    # NOTE I have tried here to restrict log(M_0)>13, which would be in line with SIMBIG's choice
    #      and initially looked like it would tighten the M_nu posterior. This didn't happen however.

    # make sure emulator is valid
    if CONSTRAIN_CONVEX_HULL and del_tess.find_simplex(theta[:NCOSMO]).item() == -1 :
        return -np.inf
    
    # LCDM part
    d = theta[:NCOSMO] - mu_LCDM
    lp_lcdm = -0.5 * np.einsum('i,ij,j->', d, covinv_LCDM, d)
    return lp_lcdm

def loglike(theta) :
    theta = (theta - norm_avg) / norm_std
    mu = model(torch.from_numpy(theta).to(dtype=torch.float32)).detach().cpu().numpy()
    d = target - mu
    return - 0.5 * np.einsum('i,ij,j->', d, covinv, d)

def logprob(theta) :
    lp = logprior(theta)
    if not np.isfinite(lp) :
        return -np.inf
    ll = loglike(theta)
    return lp + ll

if __name__ == '__main__' :

    NWALKERS = 128
    NDIM = params.shape[1]

    theta_init = np.load('vsf_mcmc_chain_wconvexhull.npy')[-1]

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
