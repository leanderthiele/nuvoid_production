from sys import argv
import re
import os
from multiprocessing import cpu_count

import numpy as np

import torch

import sbi
import sbi.inference
from sbi import utils as sbi_utils

from read_txt import read_txt
from lfi_load_posterior import load_posterior

SAMPLE_SHAPE = 20000

#if os.environ['HOSTNAME'] == 'della-gpu.princeton.edu' :
# TODO this doesn't work
#    num_cpu = 4
#else :
num_cpu = min((cpu_count(), 32))
print(f'running on {num_cpu} cpus')

NUM_WALKERS = 2*num_cpu

USE_EMCEE = True
if USE_EMCEE :
    import emcee
    from multiprocessing import Pool

filebase = '/tigress/lthiele/nuvoid_production'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_fname_base = argv[1]
try :
    fiducials_idx = int(argv[2])
except IndexError :
    fiducials_idx = None

# extract information from the model file name
match = re.search('lfi_model_v(\d*)_([a-f,0-9]{32})_([a-f,0-9]{32}).sbi', model_fname_base)
assert match is not None
version = int(match[1])
compression_hash = match[2]
model_ident = match[3]

model_fname = f'{filebase}/{model_fname_base}'

# load the settings and model from file
SETTINGS, posterior = load_posterior(model_fname, device)

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

# set the posterior to the observation
posterior = posterior.set_default_x(observation)

def logprob (theta) :
    theta = torch.from_numpy(theta.astype(np.float32)).to(device=device)
    return posterior.log_prob(theta).cpu().numpy()

if __name__ == '__main__' :

    if not USE_EMCEE :
        chain = posterior.sample(sample_shape=(SAMPLE_SHAPE,),
                                 num_workers=num_cpu, num_chains=NUM_WALKERS).cpu().numpy()
        lp = None
    else :
        theta_lo = np.array([SETTINGS['priors'][s][0] for s in SETTINGS['consider_params']])
        theta_hi = np.array([SETTINGS['priors'][s][1] for s in SETTINGS['consider_params']])
        rng = np.random.default_rng(137)
        theta_init = rng.uniform(theta_lo, theta_hi, (NUM_WALKERS, len(SETTINGS['consider_params'])))
        with Pool(num_cpu) as pool :
            sampler = emcee.EnsembleSampler(NUM_WALKERS, len(SETTINGS['consider_params']),
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

    add_info = {}
    if lp is not None :
        add_info['log_prob'] = lp
    np.savez(f'{filebase}/lfi_chain_v{version}_{compression_hash}_{model_ident}'\
             f'{f"_fid{fiducials_idx}" if fiducials_idx is not None else ""}{"_emcee" if USE_EMCEE else ""}.npz',
             chain=chain, param_names=SETTINGS['consider_params'], **add_info)
