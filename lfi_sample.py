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

chain = posterior.sample(sample_shape=(SAMPLE_SHAPE,), x=observation,
                         num_workers=cpu_count(), num_chains=cpu_count()).cpu().numpy()

np.savez(f'{filebase}/lfi_chain_v{version}_{compression_hash}_{model_ident}.npz',
         chain=chain, param_names=SETTINGS['consider_params'])
