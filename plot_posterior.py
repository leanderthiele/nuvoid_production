from sys import argv
import os.path

import numpy as np
from scipy.spatial import Delaunay

from matplotlib import pyplot as plt

import corner

filebase = '/tigress/lthiele/nuvoid_production'

chain_fname_base = argv[1]
chain_fname_base_root, _ = os.path.splitext(chain_fname_base)

chain_fname = f'{filebase}/{chain_fname_base}'
with np.load(chain_fname) as f :
    chain = f['chain']
    param_names = list(f['param_names'])
DIM = len(param_names)
assert chain.shape[-1] == DIM
chain = chain.reshape(-1, DIM)

fig = corner.corner(chain, labels=param_names, plot_datapoints=False, color='blue')

# for ii, name in enumerate(param_names) :
#     ax[ii, ii].set_title(name)

fig.savefig(f'{filebase}/{chain_fname_base_root}.pdf', bbox_inches='tight')
