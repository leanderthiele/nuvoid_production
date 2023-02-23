import numpy as np
from matplotlib import pyplot as plt

from read_txt import read_txt
from cut import Cut

plt.style.use('dark_background')

filebase = '/tigress/lthiele/nuvoid_production'
target_data = np.loadtxt(f'{filebase}/datavectors_CMASS_North.dat')

def plot_datavec (ax=None, pretty_ax=True, **plot_kwargs) :
    # plots into [-1, 1] range
    
    if ax is None :
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
    else :
        fig = None

    y = target_data / np.max(np.fabs(target_data))

    xindices = np.arange(len(target_data))
    vlines = [-0.5, ]
    part_desc = []
    for zbin in Cut.vsf_zbins :
        vlines.append(vlines[-1] + len(Cut.vsf_R))
        part_desc.append(f'vsf z{zbin}')
    for Rmin in Cut.vgplk_Rbins :
        for ell in Cut.vgplk_ell :
            vlines.append(vlines[-1] + len(Cut.vgplk_k))
            part_desc.append(f'$P_{ell}^{{vg}}$ $R_{{\\sf min}}={Rmin}$')
    for ell in Cut.plk_ell :
        vlines.append(vlines[-1] + len(Cut.plk_k))
        part_desc.append(f'$P_{ell}^{{gg}}$')

    ax.set_ylim(-1, 1)
    ax.set_xlim(0, len(target_data)-1)
    ax.plot(xindices, y, linestyle='none', marker='o')

    ymin, _ = ax.get_ylim()
    for ii, desc in enumerate(part_desc) :
        ax.text(0.5*(vlines[ii]+vlines[ii+1]), ymin, desc,
                transform=ax.transData, va='bottom', ha='center')

    if pretty_ax :
        ax.set_yscale('symlog', linthresh=1e-3)
        ax.axis('off')

    return fig, ax


if __name__ == '__main__' :
    fig, ax = plot_datavec()
    fig.savefig('_plot_datavec.png', bbox_inches='tight', transparent=True)
