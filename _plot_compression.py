import numpy as np
from matplotlib import pyplot as plt

from _plot_datavec import plot_datavec, xindices
from read_txt import read_txt
from _plot_labels import plot_labels
from _plot_style import *

compression_hashes = [
                      'faae54307696ccaff07aef77d20e1c1f',
                      # 50 percent of sims for derivatives
                      '15a4c13728710ae48cd41a03cde3dc7d',
                     ]

consider_params = [
                   'Mnu',
                   # 'hod_log_Mmin',
                  ]

filebase = '/tigress/lthiele/nuvoid_production'
version = 0

fig, ax = plot_datavec(color='grey', alpha=0.3)

for ii, compression_hash in enumerate(compression_hashes) :

    compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'
    compress_matrix = read_txt(compress_fname, 'compression matrix:')
    compress_params = read_txt(compress_fname, 'consider_params:', pyobj=True)
    compress_kwargs = read_txt(compress_fname, 'cut_kwargs:', pyobj=True)

    for p in consider_params :
        y = compress_matrix[compress_params.index(p)]
        y /= np.max(np.fabs(y))
        label = plot_labels[p]
        if len(compression_hashes) > 1 :
            label = f'{label}, $\\tt{{ {compression_hash[:4]} }}$'
        ax.plot(xindices, y, label=label, linestyle=default_linestyles[ii])

ax.legend(loc='lower left', ncol=len(consider_params), frameon=False) 

savefig(fig, f'compression_{"-".join(compression_hashes)}')
