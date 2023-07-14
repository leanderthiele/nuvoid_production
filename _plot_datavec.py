import numpy as np
from matplotlib import pyplot as plt

from read_txt import read_txt
from cut import Cut, VSF_REDGES, VGPLK_K, PLK_K

from _plot_style import *

filebase = '/tigress/lthiele/nuvoid_production'
target_data = np.loadtxt(f'{filebase}/datavector_CMASS_North_wweight.dat')
target_data_wweight = np.loadtxt(f'{filebase}/datavector_CMASS_North_wweight2.dat')
xindices = np.arange(len(target_data))
figsize = (12, 3)

# these are the best fit trial indices (averaged) for the cut data vectors
# fit to plk and vsf only
# (with kmax=0.15)
# taken from bestfit.py
bf_indices = {
              '$P^{gg}_{0,2}(k)$': (17301, 1.35), #(17301, 1.17),
              '$N_{\sf void}$': (17523, 1.03),
              '$P^{vg}_{0,2}(k)$': (23946, 1.05),
             }

vsf_R_ticks = [40, 60, ]
plk_k_ticks = [0.05, 0.15]

# where the axis labels go (hacky!)
vsf_R_labels = (55, '$R$ [Mpc/$h$]')
plk_k_labels = (0.1,'$k$ [$h$/Mpc]')

def plot_datavec (ax=None, pretty_ax=True, have_bf=False, have_xticks=False, **plot_kwargs) :
    # plots into [-1, 1] range
    
    if ax is None :
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    else :
        fig = None

    cut_vsf = Cut(use_vsf=True, use_vgplk=False, use_plk=False)
    cut_vgplk = Cut(use_vsf=False, use_vgplk=True, use_plk=False, kmin=-1, kmax=1)
    cut_plk = Cut(use_vsf=False, use_vgplk=False, use_plk=True, kmin=-1, kmax=1)

    VSF_R = 0.5*(VSF_REDGES[1:]+VSF_REDGES[:-1])
    vsf_R = np.concatenate([VSF_R, ]
                           * int(np.count_nonzero(cut_vsf.mask)/len(VSF_R)))
    vgplk_k = np.concatenate([VGPLK_K, ]
                             * int(np.count_nonzero(cut_vgplk.mask)/len(VGPLK_K)))
    plk_k = np.concatenate([PLK_K, ]
                           * int(np.count_nonzero(cut_plk.mask)/len(PLK_K)))

    def transf_datavec (x) :
        x = x.copy()
        x[cut_vgplk.mask] *= vgplk_k
        x[cut_plk.mask] *= plk_k
        x[cut_vsf.mask] *= 60
        x[cut_plk.mask] *= 3
        x /= 7100.958479575267
        return x

    y = transf_datavec(target_data)

    # FIXME
    yweight = transf_datavec(target_data_wweight)

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
            part_desc.append(f'$P_{ell}^{{vg}}(k)$\n$R^{{\sf void}}_{{\\sf min}}={Rmin}$')
    for ell in Cut.plk_ell :
        vlines.append(vlines[-1] + len(Cut.plk_k))
        part_desc.append(f'$P_{ell}^{{gg}}(k)$')

    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.5, len(target_data)-0.5)

    ax.plot(xindices, y, linestyle='none', marker='o', label='CMASS NGC OLD', **plot_kwargs)
    ax.plot(xindices, yweight, linestyle='none', marker='x', label='CMASS NGC', **plot_kwargs)

    if have_bf :
        with np.load(f'{filebase}/avg_datavectors_trials.npz') as f :
            trial_data = f['data']
        for k, (v, chisqred) in bf_indices.items() :
            ybf = transf_datavec(trial_data[v])
            ax.plot(xindices, ybf, label=f'{k} bestfit trial ($\chi^2_{{\sf red}}={chisqred:.2f}$)')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=4)

    _, ymax = ax.get_ylim()
    for ii, desc in enumerate(part_desc) :
        ax.text(0.5*(vlines[ii]+vlines[ii+1]), ymax, desc,
                transform=ax.transData, va='top', ha='center')
    for vline in vlines :
        ax.axvline(vline, color='grey')

    if pretty_ax :
        ax.axhline(0, color='grey', linestyle='dashed')
        if not have_xticks :
            ax.axis('off')
        else :
            for loc in ['left', 'right', 'top', ] :
                ax.spines[loc].set_visible(False)
            ax.set_yticks([])

    if have_xticks :
        tick_locs = []
        label_locs = []
        for t, l, b, x in [(vsf_R_ticks, vsf_R_labels, cut_vsf.mask, vsf_R),
                           (plk_k_ticks, plk_k_labels, cut_vgplk.mask, vgplk_k),
                           (plk_k_ticks, plk_k_labels, cut_plk.mask, plk_k), ] :
            start_x = xindices[b][0]
            for ii in range(len(x)-1) :
                for t_ in t :
                    if x[ii]<t_ and x[ii+1]>=t_ :
                        tick_locs.append((t_, start_x + ii + (t_ - x[ii])/(x[ii+1]-x[ii])))
                if x[ii]<l[0] and x[ii+1]>=l[0] :
                    label_locs.append((l[1], start_x + ii + (l[0] - x[ii])/(x[ii+1]-x[ii])))
        ax.set_xticks([x[1] for x in tick_locs])
        ax.set_xticklabels([str(x[0]) for x in tick_locs])
        for l, loc in label_locs :
            ax.text(loc, ax.get_ylim()[0]-0.2, l, transform=ax.transData, va='top', ha='center')


        

    return fig, ax


if __name__ == '__main__' :
    fig, ax = plot_datavec(have_bf=True, have_xticks=True)
    savefig(fig, f'datavec')
