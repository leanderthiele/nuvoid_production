from sys import argv

import numpy as np
from matplotlib import pyplot as plt

filebase = '/tigress/lthiele/nuvoid_production'

version = int(argv[1])
compression_hash = argv[2]

chain_fname = f'{filebase}/mcmc_chain_all_v{version}_{compression_hash}.npz'
logprob_fname = f'{filebase}/mcmc_logprob_all_v{version}_{compression_hash}.npy'

with np.load(chain_fname) as f :
    chain = f['chain']
    param_names = list(f['param_names'])
logprob = np.load(logprob_fname)

mnu_idx = param_names.index('Mnu')

# flatten
chain = chain.reshape(-1, chain.shape[-1])
logprob = logprob.flatten()

mnu = chain[:, mnu_idx]

mnu_edges = np.linspace(0.0, 0.6, num=32)
mnu_centers = 0.5 * (mnu_edges[1:] + mnu_edges[:-1])
bin_indices = np.digitize(mnu, mnu_edges) - 1

avg_logprob = np.empty(len(mnu_edges)-1)
for ii in range(len(mnu_edges)-1) :
    avg_logprob[ii] = np.mean(logprob[bin_indices==ii])

# normalize such that maximum at 1
avg_logprob -= np.max(avg_logprob)
avg_prob = np.exp(avg_logprob)

fig, ax = plt.subplots()
ax.plot(mnu_centers, avg_prob)
ax.set_xlabel(r'$\Sigma m_\nu$ [eV]')
ax.set_ylabel('mean likelihood')

fig.savefig(f'{filebase}/mcmc_profile_likelihood_v{version}_{compression_hash}.pdf', bbox_inches='tight')
