import numpy as np
from matplotlib import pyplot as plt

from _plot_labels import plot_labels

plt.style.use('dark_background')

filebase = '/tigress/lthiele/nuvoid_production'

def plot_ranks (ranks_base, ax, pretty=True) :
    
    with np.load(f'{filebase}/{ranks_base}') as f :
        obs_idx = f['obs_idx']
        cosmo_idx = f['cosmo_idx']
        ranks = f['ranks'] # shape [chain, parameter]
        chain_len = f['chain_len']
        param_names = list(f['param_names'])

    # normalize to [0, 1]
    xs = [ranks[:, ii].astype(float) / chain_len.astype(float) for ii in range(len(param_names))]

    # use Scott's bin width (on average)
    Nbins = int(min(np.cbrt(len(x)) / (3.5 * np.std(x)) for x in xs))
    edges = np.linspace(0, 1, num=Nbins)
    centers = 0.5 * (edges[1:] + edges[:-1])

    for param_name, x in zip(param_names, xs) :
        ax.hist(x, bins=edges, histtype='step', label=plot_labels[param_name])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)

    if pretty :
        ax.legend(loc='lower center', frameon=False)
        ax.set_xlabel('fractional position in chain')
        ax.set_ylabel('number of chains')


if __name__ == '__main__' :

    f = 'ranks_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2.npz'
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    plot_ranks(f, ax, pretty=True)

    fig.savefig('_plot_ranks.png', bbox_inches='tight', transparent=False)
