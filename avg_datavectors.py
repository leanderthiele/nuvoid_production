""" Small script to average data vectors over realizations,
    only works for trials and derivatives files.
"""

from sys import argv
import numpy as np

filebase = '/tigress/lthiele/nuvoid_production'

infile_part = argv[1]

infile = f'{filebase}/{infile_part}'
outfile = f'{filebase}/avg_{infile_part}'

with np.load(infile) as f :
    sim_idx = f['sim_idx']
    hod_hi_word = f['hod_hi_word']
    hod_lo_word = f['hod_lo_word']
    lc_idx = f['lc_idx']
    data = f['data']
    param_names = list(f['param_names'])
    params = f['params']

sim_idx_avg = []
hod_hi_word_avg = []
hod_lo_word_avg = []
data_avg = []
params_avg = []
nsims_avg = []

uniquifier = np.stack([sim_idx.astype(np.uint64), hod_hi_word, hod_lo_word], axis=-1)
_, indices, counts = np.unique(uniquifier, axis=0, return_inverse=True, return_counts=True)
unique_indices = np.unique(indices)
assert len(unique_indices) == len(counts)
for ii, c in zip(unique_indices, counts) :

    select = (indices == ii)

    select_sim_idx = sim_idx[select]
    assert all(x == select_sim_idx[0] for x in select_sim_idx)
    select_hod_hi_word = hod_hi_word[select]
    assert all(x == select_hod_hi_word[0] for x in select_hod_hi_word)
    select_hod_lo_word = hod_lo_word[select]
    assert all(x == select_hod_lo_word[0] for x in select_hod_lo_word)
    select_data = data[select]
    select_params = params[select]
    assert all(np.allclose(x, select_params[0]) for x in select_params)

    sim_idx_avg.append(select_sim_idx[0])
    hod_hi_word_avg.append(select_hod_hi_word[0])
    hod_lo_word_avg.append(select_hod_lo_word[0])
    data_avg.append(np.mean(select_data, axis=0, dtype=np.float64))
    params_avg.append(select_params[0])
    nsims_avg.append(c)

np.savez(outfile,
         sim_idx=np.array(sim_idx_avg, dtype=np.uint16),
         hod_hi_word=np.array(hod_hi_word_avg, dtype=np.uint64),
         hod_lo_word=np.array(hod_lo_word_avg, dtype=np.uint64),
         data=np.array(data_avg, dtype=np.float64),
         params=np.array(params_avg, dtype=np.float64),
         nsims=np.array(nsims_avg, dtype=np.uint64),
         param_names=param_names)
