from glob import glob
import re

import numpy as np
from matplotlib import pyplot as plt

from read_txt import read_txt
from cut import Cut

version = 0

filebase = '/tigress/lthiele/nuvoid_production'

target_data = np.loadtxt(f'{filebase}/datavector_CMASS_North.dat')

with np.load(f'{filebase}/avg_datavectors_trials.npz') as f :
    trial_data = f['data']
    trial_params = f['params']
    param_names = list(f['param_names'])

with np.load(f'{filebase}/datavectors_fiducials_v{version}.npz') as f :
    cov = np.cov(f['data'], rowvar=False)
    fid_mean = np.mean(f['data'], axis=0)

compression_files = glob(f'{filebase}/compression_v{version}_*.dat')

xindices = np.arange(len(target_data))

fig, ax = plt.subplots(nrows=2, figsize=(10,10))
ax_datavec = ax[0]
ax_residuals = ax[1]

ax_datavec.plot(xindices, target_data, linestyle='none', marker='o', label='CMASS NGC')
ax_datavec.plot(xindices, fid_mean, label=f'fiducial v{version}')

for compression_file in compression_files :
    print(compression_file)

    match = re.search(f'.*/compression_v{version}_([a-f,0-9]{{32}}).dat', compression_file)
    compression_hash = match[1]

    cut_kwargs = read_txt(compression_file, 'cut_kwargs:', pyobj=True)
    cut = Cut(**cut_kwargs)

    target_data_cut = cut.cut_vec(target_data)
    trial_data_cut = cut.cut_vec(trial_data)
    covinv_cut = np.linalg.inv(cut.cut_mat(cov))
    delta = trial_data_cut - target_data_cut[None, :]
    chisq_cut = np.einsum('ia,ab,ia->i', delta, covinv_cut, delta)
    min_idx_cut = np.argmin(chisq_cut)

    normalization = read_txt(compression_file, 'normalization:')
    compression_matrix = read_txt(compression_file, 'compression matrix:')
    target_data_cmpr = compression_matrix @ (target_data/normalization)
    trial_data_cmpr = np.einsum('ab,ib->ia', compression_matrix, trial_data/normalization[None, :])
    chisq_cmpr = np.sum((trial_data_cmpr - target_data_cmpr[None, :])**2, axis=-1)
    min_idx_cmpr = np.argmin(chisq_cmpr)

    trial_data_bf_cut = trial_data[min_idx_cut]
    trial_data_bf_cmpr = trial_data[min_idx_cmpr]

    chisq_red_bf_cut = chisq_cut[min_idx_cut] / len(target_data_cut)
    chisq_red_bf_cmpr = chisq_cmpr[min_idx_cmpr] / len(target_data_cmpr)

    delta_bf_cut = (trial_data_bf_cut - target_data) / fid_mean
    delta_bf_cmpr = (trial_data_bf_cmpr - target_data) / fid_mean

    # first do the used data
    l, _ = ax_residuals.plot(xindices[cut.mask], delta_bf_cut[cut.mask], linestyle='none', marker='o',
                             label=f'{compression_hash} cut chisq={chisq_red_bf_cut}')
    ax_residuals.plot(xindices[cut.mask], delta_bf_cmpr[cut.mask], linestyle='none', marker='s',
                      color=l.get_color(),
                      label=f'{compression hash} cmpr chisq={chisq_red_bf_cmpr}')

    # now the unused
    ax_residuals.plot(xindices[~cut.mask], delta_bf_cut[~cut.mask], linestyle='none', marker='o',
                      markerfacecolor='none', color=l.get_color())
    ax_residuals.plot(xindices[~cut.mask], delta_bf_cmpr[~cut.mask], linestyle='none', marker='s',
                      markerfacecolor='none', color=l.get_color())

ax_datavec.legend()
ax_residuals.legend()

fig.savefig('bestfit.pdf', bbox_inches='tight')
