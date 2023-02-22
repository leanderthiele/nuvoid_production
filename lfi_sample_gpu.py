import sys
from sys import argv
import re
import os
from schwimmbad import MPIPool
import emcee

import numpy as np

import torch

import sbi
import sbi.inference
from sbi import utils as sbi_utils

from read_txt import read_txt
from lfi_load_posterior import load_posterior

SAMPLE_SHAPE = 10000

NUM_WALKERS = 64

filebase = '/tigress/lthiele/nuvoid_production'

# TODO may want to use multiple GPUs here?
device = 'cuda'

try :
    fiducials_idx = int(argv[-1])
    print(f'***WARNING: Working with fiducial {fiducials_idx} instead of data!')
except (IndexError, ValueError) :
    fiducials_idx = None
model_fname_bases = argv[1:] if fiducials_idx is None else argv[1:-1]

# extract information from the model file name
version = None
compression_hash = None
model_ident = None
model_hashes = []
for model_fname_base in model_fname_bases :
    match = re.search('lfi_model_v(\d*)_([a-f,0-9]{32})_([a-f,0-9]{32}).sbi', model_fname_base)
    assert match is not None
    version_ = int(match[1])
    compression_hash_ = match[2]
    if version is None :
        version = version_
    else :
        assert version == version_
    if compression_hash is None :
        compression_hash = compression_hash_
    else :
        assert compression_hash == compression_hash_
    model_ident_ = match[3]
    model_hashes.append(model_ident_)
    if model_ident is None :
        model_ident = model_ident_
if len(model_fname_bases) > 1 :
    model_ident = f'{model_ident}_etal'

# load and compress the observation
compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'
normalization = read_txt(compress_fname, 'normalization:')
compression_matrix = read_txt(compress_fname, 'compression matrix:')

if fiducials_idx is None :
    # default case, work with real CMASS data
    observation = np.loadtxt(f'{filebase}/datavector_CMASS_North.dat')
else :
    fiducials_fname = f'{filebase}/datavectors_fiducials_v{version}.npz'
    with np.load(fiducials_fname) as f :
        observation = f['data'][fiducials_idx]
observation = compression_matrix @ (observation/normalization)

# load the settings and model from file, set posteriors to observation
priors = None
consider_params = None
posteriors = []
for model_fname_base in model_fname_bases :
    model_fname = f'{filebase}/{model_fname_base}'
    SETTINGS, posterior = load_posterior(model_fname, device)
    if priors is None :
        priors = SETTINGS['priors']
    else :
        assert priors == SETTINGS['priors']
    if consider_params is None :
        consider_params = SETTINGS['consider_params']
    else :
        assert consider_params == SETTINGS['consider_params']
    posteriors.append(posterior.set_default_x(observation))

def logprob (theta) :
    theta = torch.from_numpy(theta.astype(np.float32)).to(device=device)
    return sum(p.log_prob(theta).cpu().numpy() for p in posteriors) / len(posteriors)

with MPIPool() as pool :
    
    if not pool.is_master() :
        pool.wait()
        sys.exit(0)

    theta_lo = np.array([priors[s][0] for s in consider_params])
    theta_hi = np.array([priors[s][1] for s in consider_params])
    rng = np.random.default_rng(137)
    theta_init = rng.uniform(theta_lo, theta_hi, (NUM_WALKERS, len(consider_params)))
    sampler = emcee.EnsembleSampler(NUM_WALKERS, len(consider_params),
                                    logprob, pool=pool)
    sampler.run_mcmc(theta_init, SAMPLE_SHAPE, progress=True)
    chain = sampler.get_chain()
    lp = sampler.get_log_prob()
    acceptance_rates = sampler.acceptance_fraction
    print(f'acceptance={acceptance_rates}')
    try :
        autocorr_times  = sampler.get_autocorr_time()
        print(f'autocorr={autocorr_times}')
    except emcee.autocorr.AutocorrError :
        print(f'***WARNING: autocorr failed!')

    np.savez(f'{filebase}/lfi_chain_v{version}_{compression_hash}_{model_ident}'\
             f'{f"_fid{fiducials_idx}" if fiducials_idx is not None else ""}{"_emcee" if USE_EMCEE else ""}.npz',
             chain=chain, param_names=consider_params, model_hashes=model_hashes, log_prob=lp)
