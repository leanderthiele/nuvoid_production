from sys import argv
import re
import os
import os.path
import emcee

import numpy as np

import torch

import sbi
import sbi.inference
from sbi import utils as sbi_utils

from lfi_load_posterior import load_posterior

model_fname_base = argv[1]
start_idx = int(argv[2])

SAMPLE_SHAPE = 1000
NUM_WALKERS = 64
filebase = '/tigress/lthiele/nuvoid_production'
device = 'cuda'

# extract information from the model file name
match = re.search('lfi_model_v(\d*)_([a-f,0-9]{32})_([a-f,0-9]{32}).sbi', model_fname_base)
assert match is not None
version = int(match[1])
compression_hash = match[2]
model_ident = match[3]

# load the settings and model from file, set posteriors to observation
model_fname = f'{filebase}/{model_fname_base}'
SETTINGS, posterior = load_posterior(model_fname, device)
priors = SETTINGS['priors']
consider_params = SETTINGS['consider_params']

# load the validation set from file
validation_fname = f'{filebase}/validation_set_v{version}_{compression_hash}_{model_ident}.npz'
with np.load(validation_fname) as f :
    validation_data = f['data']
    validation_params = f['params']
    validation_sim_idx = f['sim_idx']

# store chains in this directory
outdir = f'{filebase}/coverage_chains_v{version}_{compression_hash}_{model_ident}'
os.makedirs(outdir, exist_ok=True)

def logprob (theta) :
    theta = torch.from_numpy(theta.astype(np.float32)).to(device=device)
    return posterior.potential(theta).cpu().numpy()

# loop until time is up
for obs_idx in range(start_idx, 100000) :
    
    x = validation_data[obs_idx]
    p = validation_params[obs_idx]
    cosmo_idx = validation_sim_idx[obs_idx]

    outfname = f'{outdir}/chain_{obs_idx}_cosmo{cosmo_idx}.npz'
    if os.path.isfile(outfname) :
        continue

    print(f'working on observation index {obs_idx}')

    posterior = posterior.set_default_x(x)

    theta_lo = np.array([priors[s][0] for s in consider_params])
    theta_hi = np.array([priors[s][1] for s in consider_params])
    rng = np.random.default_rng(137+obs_idx)
    theta_init = rng.uniform(theta_lo, theta_hi, (NUM_WALKERS, len(consider_params)))
    sampler = emcee.EnsembleSampler(NUM_WALKERS, len(consider_params),
                                    logprob,
                                    moves=emcee.moves.StretchMove(a=5.0),
                                    # pool=pool,
                                   )
    sampler.run_mcmc(theta_init, SAMPLE_SHAPE, progress=False)
    chain = sampler.get_chain()
    lp = sampler.get_log_prob()
    acceptance_rates = sampler.acceptance_fraction
    print(f'acceptance={acceptance_rates}')
    try :
        autocorr_times  = sampler.get_autocorr_time()
        print(f'autocorr={autocorr_times}')
    except emcee.autocorr.AutocorrError :
        print(f'***WARNING: autocorr failed!')

    np.savez(outfname,
             chain=chain.astype(np.float32),
             log_prob=lp.astype(np.float32),
             param_names=consider_params,
             real_params=p)
