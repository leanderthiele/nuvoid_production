from sys import argv
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.spatial import Delaunay

import torch
import emcee

from mlp import MLP
from read_txt import read_txt

MCMC_STEPS = 20000

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
cov_LCDM = np.loadtxt(mu_cov_fname, skiprows=3)
covinv_LCDM = np.linalg.inv(cov_LCDM)

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
in_tess_LCDM = lambda x: del_tess_LCDM.find_simplex(x).item() != -1

# get the uniform HOD priors
min_HOD = np.min(p_HOD, axis=0)
max_HOD = np.max(p_HOD, axis=0)

def logprior (theta) :
    theta_LCDM = theta[:NCOSMO]
    Mnu = theta[NCOSMO]
    theta_HOD = theta[NCOSMO+1:]
    if not np.all((theta_HOD<=max_HOD)*(theta_HOD>=min_HOD)) :
        return -np.inf
    if not in_tess_LCDM(theta_LCDM) :
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

    print('Starting constructing starting positions')
    theta_init = np.empty((NWALKERS, NDIM))
    rng = np.random.default_rng(42)
    for ii in range(NWALKERS) :
        while True :
            rnd_LCDM = rng.multivariate_normal(mu_LCDM, cov=cov_LCDM)
            if in_tess_LCDM(rnd_LCDM) :
                theta_init[ii, :NCOSMO] = rnd_LCDM
                break
        rnd_Mnu = rng.uniform(0.0, 0.6)
        theta_init[ii, NCOSMO] = rnd_Mnu
        rnd_HOD = rng.uniform(min_HOD, max_HOD)
        theta_init[ii, NCOSMO+1:] = rnd_HOD

    print('Finished constructing starting positions')

    print(f'Running on {cpu_count()} CPUs')
    with Pool() as pool :
        sampler = emcee.EnsembleSampler(NWALKERS, NDIM, logprob, pool=pool)

        sampler.run_mcmc(theta_init, MCMC_STEPS, progress=True)

        chain = sampler.get_chain(thin=30, discard=MCMC_STEPS//5)
        np.save(f'{filebase}/mcmc_chain_v{version}_{compression_hash}.npy', chain)
        
        chain_all = sampler.get_chain()
        np.save(f'{filebase}/mcmc_chain_all_v{version}_{compression_hash}.npy', chain_all)

        logprob = sampler.get_log_prob()
        np.save(f'{filebase}/mcmc_logprob_all_v{version}_{compression_hash}.npy', logprob)

        acceptance_rates = sampler.acceptance_fraction
        print(f'acceptance={acceptance_rates}')

        autocorr_times = sampler.get_autocorr_time()
        print(f'autocorr={autocorr_times}')
