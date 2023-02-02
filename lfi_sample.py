from sys import argv
import io
import pickle
import re
from multiprocessing import cpu_count

import numpy as np

import torch

import sbi
import sbi.inference
from sbi import utils as sbi_utils

from read_txt import read_txt

SAMPLE_SHAPE = 20000
NUM_WALKERS = cpu_count()

USE_EMCEE = False
if USE_EMCEE :
    import emcee
    from multiprocessing import Pool

filebase = '/tigress/lthiele/nuvoid_production'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_fname_base = argv[1]

# extract information from the model file name
match = re.search('lfi_model_v(\d*)_([a-f,0-9]{32})_([a-f,0-9]{32}).sbi', model_fname_base)
assert match is not None
version = int(match[1])
compression_hash = match[2]
model_ident = match[3]

model_fname = f'{filebase}/{model_fname_base}'

class Unpickler(pickle.Unpickler) :
    """ small utility to load cross-device """
    def find_class (self, module, name) :
        if module == 'torch.storage' and name == '_load_from_bytes' :
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else :
            return super().find_class(module, name)

# load the settings and model from file
settings_str = b''
with open(model_fname, 'rb') as f :
    while True :
        # scan until we reach newline, which indicates start of the pickled model
        c = f.read(1)
        if c == b'\n' :
            break
        settings_str += c
    posterior = Unpickler(f).load()

SETTINGS = eval(settings_str)

# fix some stuff
posterior._device = device
posterior.potential_fn.device = device

# load and compress the observation
compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'
normalization = read_txt(compress_fname, 'normalization:')
compression_matrix = read_txt(compress_fname, 'compression matrix:')
observation = np.loadtxt(f'{filebase}/datavector_CMASS_North.dat')
observation = compression_matrix @ (observation/normalization)

# set the posterior to the observation
posterior = posterior.set_default_x(observation)

def logprob (theta) :
    theta = torch.from_numpy(theta.astype(np.float32)).to(device=device)
    return posterior.log_prob(theta).cpu().numpy()

if __name__ == '__main__' :

    if not USE_EMCEE :
        chain = posterior.sample(sample_shape=(SAMPLE_SHAPE,),
                                 num_workers=cpu_count(), num_chains=NUM_WALKERS).cpu().numpy()
        lp = None
    else :
        theta_lo = np.array([SETTINGS['priors'][s][0] for s in SETTINGS['consider_params']])
        theta_hi = np.array([SETTINGS['priors'][s][1] for s in SETTINGS['consider_params']])
        rng = np.random.default_rng(137)
        theta_init = rng.uniform(theta_lo, theta_hi, (NUM_WALKERS, len(SETTINGS['consider_params'])))
        with Pool() as pool :
            sampler = emcee.EnsembleSampler(NUM_WALKERS, len(SETTINGS['consider_params']),
                                            logprob, pool=pool)
            sampler.run_mcmc(theta_init, SAMPLE_SHAPE, progress=True)
            chain = sampler.get_chain()
            lp = sampler.get_log_prob()
            acceptance_rates = sampler.acceptance_fraction
            print(f'acceptance={acceptance_rates}')
            autocorr_times  = sampler.get_autocorr_time()
            print(f'autocorr={autocorr_times}')

    add_info = {}
    if lp is not None :
        add_info['log_prob'] = lp
    np.savez(f'{filebase}/lfi_chain_v{version}_{compression_hash}_{model_ident}{"_emcee" if USE_EMCEE else ""}.npz',
             chain=chain, param_names=SETTINGS['consider_params'], **add_info)
