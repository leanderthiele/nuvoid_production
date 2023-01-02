from collections import OrderedDict
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.spatial import Delaunay
from scipy.special import gammaln, xlogy

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

MCMC_STEPS = 100000

# pairs mean, sigma
ADD_PRIOR = {'hod_transfP1': (0.0, 0.1),
             'hod_abias': (0.0, 0.1),
             'hod_log_Mmin': (12.9, 0.01),
             'hod_sigma_logM': (0.4, 0.01),
             'hod_log_M0': (14.4, 0.1),
             'hod_log_M1': (14.4, 0.1),
             'hod_alpha': (0.6, 0.01),
             'hod_transf_eta_cen': (6.0, 0.1),
             'hod_transf_eta_sat': (-0.5, 0.01),
             'hod_mu_Mmin': (-5.0, 0.1),
             'hod_mu_M1': (10.0, 1.0),
            }

with np.load('/tigress/lthiele/collected_vgplk.npz') as f :
    param_names = list(f['param_names'])
    params = f['params']
    Rmin = f['Rmin']

with np.load('hists.npz') as f :
    param_names_ = list(f['param_names'])
    assert all(a==b for a, b in zip(param_names, param_names_))
    hists = f['hists']
    vsf_shape = hists.reshape(hists.shape[0], -1).shape[-1]
    target_vsf = f['hist_cmass'].flatten()

with np.load('/tigress/lthiele/cov_vgplk.npz') as f :
    cov = f['cov'] # shape [Rmin, ellxk]
    k_indices = f['k_indices']
    assert np.allclose(Rmin, f['Rmin'])
rmin_idx = np.where(Rmin == RMIN)[0][0]
cov = cov[rmin_idx, ...]
# remove the quadrupole if requested
if not USE_QUADRUPOLE :
    cov = cov[:len(k_indices), :len(k_indices)]
covinv = np.linalg.inv(cov)

with np.load('vgplk_norm.npz') as f :
    vgplk_norm_avg = f['avg']
    vgplk_norm_std = f['std']
with np.load('vsf_mlp_norm.npz') as f :
    vsf_norm_avg = f['avg']
    vsf_norm_std = f['std']

with np.load('/tigress/lthiele/cmass_vgplk.npz') as f :
    target_vgplk = f[f'p0k_Rmin{RMIN}'][k_indices]
    if USE_QUADRUPOLE :
        target_vgplk = np.concatenate([target_vgplk, f[f'p2k_Rmin{RMIN}'][k_indices]])

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

# additional prior, only for testing
mu_add = np.zeros(params.shape[-1])
varinv_add = np.zeros(params.shape[-1])
for ii, name in enumerate(param_names) :
    if name in ADD_PRIOR :
        mu, sigma = ADD_PRIOR[name]
        mu_add[ii] = mu
        varinv_add[ii] = 1.0/sigma**2

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

model_vgplk = MLP(params.shape[-1], (1+USE_QUADRUPOLE)*len(k_indices), Nlayers=4, Nhidden=512, out_positive=False)
model_vgplk.load_state_dict(torch.load('vgplk_mlp.pt', map_location='cpu'))

model_vsf = MLP(params.shape[-1], vsf_shape, Nlayers=8, Nhidden=512, out_positive=True)
model_vsf.load_state_dict(torch.load('vsf_mlp.pt', map_location='cpu'))

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

    # additional part
    lp_add = -0.5 * np.sum((theta - mu_add)**2 * varinv_add)

    return lp_lcdm + lp_add

def loglike(theta) :
    theta_vsf = (theta - vsf_norm_avg) / vsf_norm_std
    mu_vsf = model_vsf(torch.from_numpy(theta_vsf).to(dtype=torch.float32)).detach().cpu().numpy()
    l_vsf = np.sum(xlogy(target_vsf, mu_vsf) - mu_vsf - gammaln(1.0+target_vsf))

    theta_vgplk = (theta - vgplk_norm_avg) / vgplk_norm_std
    mu_vgplk = model_vgplk(torch.from_numpy(theta_vgplk).to(dtype=torch.float32)).detach().cpu().numpy()
    d = target_vgplk - mu_vgplk
    l_vgplk = - 0.5 * np.einsum('i,ij,j->', d, covinv, d)
    return l_vsf + l_vgplk

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
        np.save('vsfvgplk_mcmc_chain.npy', chain)

        acceptance_rates = sampler.acceptance_fraction
        print(f'acceptance={acceptance_rates}')

        autocorr_times = sampler.get_autocorr_time()
        print(f'autocorr={autocorr_times}')
