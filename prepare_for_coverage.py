from sys import argv
import re

import numpy as np

from lfi_load_posterior import load_posterior

filebase = '/tigress/lthiele/nuvoid_production'
model_fname_base = argv[1]

match = re.search('lfi_model_v(\d*)_([a-f,0-9]{32})_([a-f,0-9]{32}).sbi', model_fname_base)
version = int(match[1])
compression_hash = match[2]
model_ident = match[3]

model_fname = f'{filebase}/{model_fname_base}'
data_fname = f'{filebase}/datavectors_trials.npz'
fiducials_fname = f'{filebase}/datavectors_fiducials_v0.npz'
compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'
out_fname = f'{filebase}/validation_set_v{version}_{compression_hash}_{model_ident}.npz'

SETTINGS, _ = load_posterior(model_fname, None, need_posterior=False)

if 'val_contiguous' in SETTINGS :
    val_sim_idx = sum((list(range(s, e+1)) for s, e in SETTINGS['val_contiguous']), start=[])
else :
    val_sim_idx = [ 93, 21, 80, 8, 67, 126, 73, 39, 98, 26, 85, 125, 34, ]

normalization = read_txt(compress_fname, 'normalization:')
compression_matrix = read_txt(compress_fname, 'compression matrix:')
with np.load(data_fname) as f :
    data = f['data'].astype(np.float64)
    param_names = list(f['param_names'])
    params = f['params']
    sim_idx = f['sim_idx']
data = np.einsum('ab,ib->ia', compression_matrix, data/normalization[None, :])
nbefore = len(params)
for k, (lo, hi) in SETTINGS['priors'].items() :
    param_idx = param_names.index(k)
    mask = (params[:, param_idx]>=lo) * (params[:, param_idx]<=hi)
    params = params[mask]
    data = data[mask]
    sim_idx = sim_idx[mask]
nafter = len(params)
print(f'Retaining {nafter/nbefore * 100} percent of samples after prior')

param_indices = [param_names.index(s) for s in SETTINGS['consider_params']]
params = params[:, param_indices]

if 'chisq_max' in SETTINGS :
    with np.load(fiducials_fname) as f :
        fid_mean = np.mean(f['data'], axis=0)
    fid_mean = compression_matrix @ (fid_mean/normalization)
    chisq = np.sum((data - fid_mean[None, :])**2, axis=-1) / data.shape[-1]
    mask = chisq < SETTINGS['chisq_max']
    print(f'After chisq_max cut, retaining {np.count_nonzero(mask)/len(mask)*100:.2f} percent of samples')
    data = data[mask]
    params = params[mask]
    sim_idx = sim_idx[mask]

validation_mask = np.array([idx in val_sim_idx for idx in sim_idx], dtype=bool)

data = data[validation_mask]
params = params[validation_mask]
sim_idx = sim_idx[validation_mask]

np.savez(out_fname, data=data, params=params, sim_idx=sim_idx)
