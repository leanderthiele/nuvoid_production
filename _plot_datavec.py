import numpy as np
from matplotlib import pyplot as plt

from read_txt import read_txt
from cut import Cut, VGPLK_K, PLK_K

plt.style.use('dark_background')

filebase = '/tigress/lthiele/nuvoid_production'
target_data = np.loadtxt(f'{filebase}/datavector_CMASS_North.dat')

def plot_datavec (ax=None, pretty_ax=True, **plot_kwargs) :
    # plots into [-1, 1] range
    
    if ax is None :
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
    else :
        fig = None

    cut_vsf = Cut(use_vsf=True, use_vgplk=False, use_plk=False)
    cut_vgplk = Cut(use_vsf=False, use_vgplk=True, use_plk=False, kmin=-1, kmax=1)
    cut_plk = Cut(use_vsf=False, use_vgplk=False, use_plk=True, kmin=-1, kmax=1)

    vgplk_k = np.concatenate([VGPLK_K, ] * int(np.count_nonzero(cut_vgplk.mask)/len(VGPLK_K)))
    plk_k = np.concatenate([PLK_K, ] * int(np.count_nonzero(cut_plk.mask)/len(PLK_K)))

    y = target_data.copy()
    y[cut_vgplk.mask] *= vgplk_k
    y[cut_plk.mask] *= plk_k
    y[cut_vsf.mask] *= 60
    y[cut_plk.mask] *= 3

    y /= np.max(np.fabs(y))

    xindices = np.arange(len(target_data))
    vlines = [-0.5, ]
    part_desc = []
    for zbin in Cut.vsf_zbins :
        vlines.append(vlines[-1] + len(Cut.vsf_R))
        if zbin == 0 :
            zstr = '<0.53'
        else :
            zstr = '>0.53'
        part_desc.append(f'$N_{{void}}$\n$z{zstr}$')
    for Rmin in Cut.vgplk_Rbins :
        for ell in Cut.vgplk_ell :
            vlines.append(vlines[-1] + len(Cut.vgplk_k))
            part_desc.append(f'$P_{ell}^{{vg}}$\n$R^{{\sf void}}_{{\\sf min}}={Rmin}$')
    for ell in Cut.plk_ell :
        vlines.append(vlines[-1] + len(Cut.plk_k))
        part_desc.append(f'$P_{ell}^{{gg}}$')

    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.5, len(target_data)-0.5)
    ax.plot(xindices, y, linestyle='none', marker='o')

    _, ymax = ax.get_ylim()
    for ii, desc in enumerate(part_desc) :
        ax.text(0.5*(vlines[ii]+vlines[ii+1]), ymax, desc,
                transform=ax.transData, va='top', ha='center')
    for vline in vlines :
        ax.axvline(vline, color='grey')

    if pretty_ax :
        ax.axis('off')
        ax.axhline(0, color='grey', linestyle='dashed')

    return fig, ax


if __name__ == '__main__' :
    fig, ax = plot_datavec()
    fig.savefig('_plot_datavec.png', bbox_inches='tight', transparent=False)
