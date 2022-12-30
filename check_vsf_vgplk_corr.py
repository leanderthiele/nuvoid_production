import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

KMIN = 0.025
KMAX = 0.1

Rmins = [40, ]
Nbins = [16, ]

fig, ax = plt.subplots(ncols=len(Nbins), nrows=len(Rmins),
                       figsize=(5*len(Nbins),5*len(Rmins)))
try :
    ax[0]
except TypeError :
    ax = np.array([ax,])

try :
    ax[0,0]
except IndexError :
    ax = np.array([ax,])

def remove_frame(axis) :
    for s in ['top', 'bottom', 'right', 'left'] :
        axis.spines[s].set_visible(False)
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])

for row, Rmin in enumerate(Rmins) :
    for col, Nbin in enumerate(Nbins) :
        
        print(row, col)
        a = ax[row, col]

        fvgplk = np.load('fiducial_vgplk.npz')
        k = fvgplk['k']
        Rmin_ = fvgplk['Rmin']
        seeds = list(map(lambda s: int(s.split('_')[1]),
                         list(filter(lambda s: 'seed' in s, list(fvgplk.keys())))))

        min_idx = np.argmin(np.fabs(k-KMIN))
        max_idx = np.argmin(np.fabs(k-KMAX))
        rmin_idx = np.argmin(np.fabs(Rmin_-Rmin))

        all_vgplk = np.stack([fvgplk[f'seed_{seed}'] for seed in seeds], axis=0)
        all_vgplk = all_vgplk[..., min_idx : max_idx+1][:, :, rmin_idx, :]
        all_vgplk = all_vgplk.reshape(*all_vgplk.shape[:2], -1)

        fvsf = np.load(f'fiducial_vsfs_{Nbin}bins.npz')
        seeds_ = list(map(lambda s: int(s.split('_')[1]),
                          list(filter(lambda s: 'seed' in s, list(fvsf.keys())))))
        assert not set(seeds).symmetric_difference(set(seeds_))

        all_vsf = np.stack([fvsf[f'seed_{seed}'] for seed in seeds], axis=0)

        all_data = np.concatenate([all_vsf, all_vgplk], axis=-1)

        # flatten out the realization vs augmentation axes
        all_data = all_data.reshape(-1, all_data.shape[-1])

        # choose the ones where we have measuremnts
        select = np.where(np.all(np.isfinite(all_data), axis=-1))[0]
        all_data = all_data[select]
        print(all_data.shape)

        cov = np.cov(all_data, rowvar=False)
        corr = cov / np.sqrt(np.diagonal(cov)[:, None] * np.diagonal(cov)[None, :])

        im = a.matshow(corr, vmin=-1, vmax=1, cmap='seismic',
                       extent=(0, len(corr), len(corr), 0))
        divider = make_axes_locatable(a)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        a.set_title(f'VSF Nbins={Nbin}, vgplk Rmin={Rmin}')
        remove_frame(a)

        block_starts = [0, all_vsf.shape[-1]/2, all_vsf.shape[-1], all_vsf.shape[-1]+all_vgplk.shape[-1]/2, ]
        block_ends = [all_vsf.shape[-1]/2, all_vsf.shape[-1], all_vsf.shape[-1]+all_vgplk.shape[-1]/2, ] 
        block_lens = [all_vsf.shape[-1]/2, all_vsf.shape[-1]/2, all_vgplk.shape[-1]/2, all_vgplk.shape[-1]/2, ]
        texts = ['$z<0.53$', '$z>0.53$', '$\ell=0$', '$\ell=2$', ]
        for x in block_ends :
            for s in ['axhline', 'axvline', ] :
                getattr(a, s)(x, color='grey')
        for x, l, t in zip(block_starts, block_lens, texts) :
            a.text(x+l/2, x+1, t, transform=a.transData, ha='center', va='top')


fig.suptitle('Complete correlation matrices')
fig.savefig('check_vsf_vgplk_corr.pdf', bbox_inches='tight')
