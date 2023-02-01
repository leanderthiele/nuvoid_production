from sys import argv
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.spatial import Delaunay

import torch

from mlp import MLP
from read_txt import read_txt

filebase = '/tigress/lthiele/nuvoid_production'

version = int(argv[1])
compression_hash = argv[2]

# load the model from disk and initialize it
f = torch.load(f'{filebase}/mlp_v{version}_{compression_hash}.pt', map_location='cpu')
model_state = f['model_state']
model_meta = f['model_meta']
input_params = f['input_params']
params_avg = f['params_avg']
params_std = f['params_std']

model = MLP(**model_meta)
model.load_state_dict(model_state)
model.eval()

# load the target from disk and compress it
compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'
normalization = read_txt(compress_fname, 'normalization:')
compression_matrix = read_txt(compress_fname, 'compression matrix:')
target = compression_matrix @ (np.loadtxt(f'{filebase}/datavector_CMASS_North.dat')/normalization)

# construct the LCDM prior -- TODO maybe it is necessary to include Mnu in the tesselation???
NCOSMO = 5
prior_params = ['Obh2', 'Och2', 'theta', 'logA', 'ns', ]
assert prior_params == input_params[:NCOSMO]
mu_cov_fname = f'{filebase}/mu_cov_plikHM_TTTEEE_lowl_lowE.dat'
mu_LCDM = np.loadtxt(mu_cov_fname, max_rows=1)
covinv_LCDM = np.linalg.inv(np.loadtxt(mu_cov_fname, skiprows=3))

# load some information about the points where we have samples
with np.load(f'{filebase}/avg_datavectors_trials.npz') as f :
    sim_idx = f['sim_idx']
    p = f['params']
    param_names = list(f['param_names'])
param_indices = [param_names.index(s) for s in input_params]
p = p[:, param_indices]
_, first_idx = np.unique(sim_idx, return_index=True)
p_LCDM = p[first_idx, :][:, :NCOSMO]
p_HOD = p[:, np.array(['hod' in s for s in input_params], dtype=bool)]

# construct the Delaunay triangulation of LCDM
del_tess_LCDM = Delaunay(p_LCDM)

# get the uniform HOD priors
min_HOD = np.min(p_HOD, axis=0)
max_HOD = np.max(p_HOD, axis=0)

def logprior (theta) :
    theta_LCDM = theta[:NCOSMO]
    Mnu = theta[NCOSMO]
    theta_HOD = theta[NCOSMO+1:]
    if not np.all((theta_HOD<=max_HOD)*(theta_HOD>=min_HOD)) :
        return -np.inf
    if del_tess_LCDM.find_simplex(theta_LCDM).item() == -1 :
        return -np.inf
    if Mnu<0 or Mnu>0.6 :
        return -np.inf
    d = theta_LCDM - mu_LCDM
    lp_lcdm = -0.5 * np.einsum('a,ab,b->', d, covinv_LCDM, d)
    return lp_lcdm

def loglike (theta) :
    torch_theta = torch.from_numpy(theta.copy()).to(dtype=torch.float32)
    torch_theta = (torch_theta - params_avg) / params_std
    mu = model(torch_theta).detach().cpu().numpy()
    return (mu - target)**2 # covariance should be unity

def logprob (theta) :
    lp = logprior(theta)
    if not np.isfinite(lp) :
        return -np.inf
    ll = loglike(theta)
    return lp + ll

if __name__ == '__main__' :

    NWALKERS = 128
    NDIM = model.Nin

    theta_init = np.load('vsf_mcmc_chain_wconvexhull.npy')[-1]

    print(f'Running on {cpu_count()} CPUs')
    with Pool() as pool :
        sampler = emcee.EnsembleSampler(NWALKERS, NDIM, logprob, pool=pool)

        sampler.run_mcmc(theta_init, MCMC_STEPS, progress=True)

        chain = sampler.get_chain(thin=30, discard=MCMC_STEPS//5)
        np.save('vgplk_mcmc_chain.npy', chain)

        acceptance_rates = sampler.acceptance_fraction
        print(f'acceptance={acceptance_rates}')

        autocorr_times = sampler.get_autocorr_time()
        print(f'autocorr={autocorr_times}')
