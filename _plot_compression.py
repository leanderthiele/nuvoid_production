import numpy as np
from matplotlib import pyplot as plt

from _plot_datavec import plot_datavec, xindices
from read_txt import read_txt
from _plot_labels import plot_labels

compression_hash = 'faae54307696ccaff07aef77d20e1c1f'
consider_params = ['Mnu', 'hod_log_Mmin', ]

filebase = '/tigress/lthiele/nuvoid_production'
version = 0

compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'
compress_matrix = read_txt(compress_fname, 'compression matrix:')
compress_params = read_txt(compress_fname, 'consider_params:', pyobj=True)
compress_kwargs = read_txt(compress_fname, 'cut_kwargs:', pyobj=True)

fig, ax = plot_datavec(pretty_ax=True)

for p in consider_params :
    y = compress_matrix[compress_params.index(p)]
    y /= np.max(np.fabs(y))
    ax.plot(xindices, y, label=plot_labels[p])

ax.legend(loc='lower left', ncol=len(consider_params), frameon=False) 

fig.savefig(f'_plot_compression_{compression_hash}.png', bbox_inches='tight', transparent=False)
