from sys import argv
from glob import glob
import re

import numpy as np

filebase = '/tigress/lthiele/nuvoid_production'
discard = 100 # burn in

outdir_base = argv[1]

match = re.search('coverage_chains_v(\d*)_([a-f,0-9]{32})_([a-f,0-9]{32})', outdir_base)
version = int(match[1])
compression_hash = match[2]
model_ident = match[3]

param_names = None
obs_idx = []
cosmo_idx = []
ranks = []
chain_len = []

chain_fnames = glob(f'{filebase}/{outdir_base}/chain_*.npz')

for chain_fname in chain_fnames :
    
    match = re.search('chain_(\d*)_cosmo(\d*).npz', chain_fname)
    obs_idx_  = int(match[1])
    cosmo_idx_ = int(match[2])
    
    with np.load(chain_fname) as f :
        param_names_ = list(f['param_names'])
        chain = f['chain']
        real_params = f['real_params']

    if param_names is None :
        param_names = param_names_
    else :
        assert param_names == param_names_
    chain = chain[discard:].reshape(-1, chain.shape[-1])

    ranks_ = np.sum(real_params[None, :] > chain, axis=0, dtype=int)
    
    obs_idx.append(obs_idx_)
    cosmo_idx.append(cosmo_idx_)
    ranks.append(ranks_)
    chain_len.append(len(chain))

np.savez(f'{filebase}/ranks_v{version}_{compression_hash}_{model_ident}.npz',
         obs_idx=np.array(obs_idx), cosmo_idx=np.array(cosmo_idx),
         ranks=np.array(ranks), chain_len=np.array(chain_len),
         param_names=np.array(param_names))
