import subprocess

import math
import numpy as np
from matplotlib import pyplot as plt

from _plot_labels import plot_labels
from _plot_fiducials import fid
from _plot_style import *

cosmo_idx = []
param_names = None
params = []
for ii in range(150) :
    result = subprocess.run(f'./mysql_driver get_cosmo {ii}', shell=True, capture_output=True)
    if result.returncode != 0 :
        continue
    cosmo_idx.append(ii)
    lines = result.stdout.decode().split()
    _param_names = []
    _params = []
    for line in lines :
        k, v = line.split('=')
        _param_names.append(k)
        _params.append(float(v))
    if param_names is None :
        param_names = _param_names
    else :
        assert param_names == _param_names, (param_names, _param_names)
    params.append(np.array(_params))
params = np.array(params)
mnu_idx = param_names.index('Mnu')
        
def make_plot (consider_params, name) :
    
    D = len(consider_params)

    fig, ax = plt.subplots(nrows=D-1, ncols=D-1, figsize=(5,5),
                           gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

    for row in range(1, D) :
        row_idx = param_names.index(consider_params[row])
        for col in range(row) :
            col_idx = param_names.index(consider_params[col])
            ax[row-1, col].scatter(params[:, col_idx], params[:, row_idx],
                                   s=3, c=params[:, mnu_idx])
            ax[row-1, col].scatter([fid[consider_params[col]],], [fid[consider_params[row]],],
                                   s=20, c='red', marker='x')

    for row in range(1, D) :
        ax[row-1, 0].set_ylabel(plot_labels[consider_params[row]])
        for col in range(1, D-1) :
            ax[row-1, col].set_yticklabels([])
    for col in range(D-1) :
        ax[-1, col].set_xlabel(plot_labels[consider_params[col]])
        for row in range(0, D-2) :
            ax[row, col].set_xticklabels([])

    for a in ax.flatten() :
        if not a.collections :
            a.axis('off')
        else :
            a.tick_params(axis='x', labelrotation=45)

    fig.align_labels()
    savefig(fig, f'sampled_params_{name}')

make_plot(['Mnu', 'Och2', 'Obh2', 'logA', 'ns', 'theta', ], 'CMB')
make_plot(['Mnu', 'Om', 'Ob', 'sigma8', 'ns', 'h', ], 'phys')
