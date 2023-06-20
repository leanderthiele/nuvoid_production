import numpy as np
from matplotlib import pyplot as plt

from _plot_labels import plot_labels, nounit
from _plot_style import *

filebase = '/tigress/lthiele/nuvoid_production'
Nbins = 50

def plot_coverage (fname_base, ax, pretty=True) :
    
    with np.load(f'{filebase}/{fname_base}') as f :
        param_names = list(f['param_names'])
        obs_idx = f['obs_idx']
        cosmo_idx = f['cosmo_idx']
        oma = f['oneminusalpha']

    edges = np.linspace(0, 1, num=Nbins+1)

    for param_name, x in zip(param_names, oma.T) :
        x = x[x>=0]
        coverage = np.array([np.count_nonzero(x<e) for e in edges]) / len(x)
        ax.plot(edges, coverage, label=nounit(plot_labels[param_name]),
                marker='o' if param_name=='Mnu' else None)

    ax.axline((0, 0), slope=1, color='grey', linestyle='dashed')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if pretty :
        ax.legend(loc='center right', frameon=False)
        ax.set_xlabel('confidence level')
        ax.set_ylabel('empirical coverage')
        ax.text(0.05, 0.95, 'underconfident', va='top', ha='left', transform=ax.transAxes)
        ax.text(0.95, 0.05, 'overconfident', va='bottom', ha='right', transform=ax.transAxes)


if __name__ == '__main__' :
    
    f = 'oneminusalpha_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2.npz'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.5))
    plot_coverage(f, ax, pretty=True)

    savefig(fig, f'coverage')
