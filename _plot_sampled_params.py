import subprocess

import math
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('dark_background')

labels = {
          'Mnu': r'$\Sigma m_\nu$',
          'Om': r'$\Omega_m$',
          'Ob': r'$\Omega_b$',
          'h': r'$h$',
          'ns': r'$n_s$',
          '1e9As': r'$10^9 A_s$',
          'sigma8': r'$\sigma_8$',
          'Oc': r'$\Omega_{\sf cdm}$',
          'On': r'$\Omega_\nu$',
          'S8': r'$S_8$',
          'logA': r'$\log 10^{10} A_s$',
          'theta': r'$\theta_{\sf MC}$',
          'Obh2': r'$\Omega_b h^2$',
          'Och2': r'$\Omega_{\sf cdm} h^2$',
         }

fid = {'Mnu': 0.1,
       'Om': 0.3219034692307693,
       'Ob': 0.049976779230769236,
       'h': 0.66908459,
       'ns': 0.9651028379487179,
       '1e9As': 2.0971604141485533,
       'logA': 3.043169338959259,
       'Oc': 0.26952822,
       'On': 0.00239847,
       'sigma8': 0.80562700,
       'Obh2': 0.022373314089749925,
       'Och2': 0.12066082718669024,
       'S8': 0.8345189714099627,
       'theta': 1.041118752752947,
      }

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

    fig, ax = plt.subplots(nrows=D-1, ncols=D-1, figsize=(8,8),
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
        ax[row-1, 0].set_ylabel(labels[consider_params[row]])
        for col in range(1, D-1) :
            ax[row-1, col].set_yticklabels([])
    for col in range(D-1) :
        ax[-1, col].set_xlabel(labels[consider_params[col]])
        for row in range(0, D-2) :
            ax[row, col].set_xticklabels([])

    for a in ax.flatten() :
        if not a.collections :
            a.axis('off')

    fig.savefig(f'_plot_sampled_params_{name}.png', bbox_inches='tight', transparent=True)

make_plot(['Mnu', 'Och2', 'Obh2', 'logA', 'ns', 'theta', ], 'CMB')
make_plot(['Mnu', 'Om', 'Ob', 'sigma8', 'ns', 'h', ], 'phys')
