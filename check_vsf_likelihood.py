import sys
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

NBINS = int(argv[1])

f = np.load(f'fiducial_vsfs_{NBINS}bins.npz')
Redges = f['Redges']
zedges = f['zedges']

cmass_z, cmass_R = np.loadtxt('/tigress/lthiele/boss_dr12/voids/sample_test/sky_positions_central_test.out',
                              usecols=(2,3), unpack=True)
cmass_vsf = np.histogram2d(cmass_z, cmass_R, bins=[zedges, Redges])[0].flatten()

seeds = list(map(lambda s: int(s.split('_')[1]), list(filter(lambda s: 'seed' in s, list(f.keys())))))
all_hists = np.stack([f[f'seed_{seed}'] for seed in seeds], axis=0)
print(all_hists.shape) # [realization, augmentation, index]

mean = np.nanmean(all_hists.reshape(-1, all_hists.shape[-1]), axis=0)

# downsample augmentations to same number as realizations so we have identical stochasticity
N_augments = len(seeds)
good_augments = np.where(np.all(np.isfinite(all_hists[..., 0]), axis=0))[0]
assert(len(good_augments)>N_augments)
rng = np.random.default_rng(42)
use_augments = rng.choice(good_augments, N_augments, replace=False)

all_hists = all_hists[:, use_augments, :]

assert np.all(np.isfinite(all_hists))

# this is the approximate answer
stds_by_realization = np.array([np.nanstd(h, axis=0) for h in all_hists])

# this is the correct (reference answer)
stds_by_augmentation = np.array([np.nanstd(all_hists[:, ii, :], axis=0) for ii in range(N_augments)])

# approximate answer (average over augmentations)
vsfs_by_realization = np.array([np.nanmean(h, axis=0) for h in all_hists])

# correct answer (average over realizations)
vsfs_by_augmentation = np.array([np.nanmean(all_hists[:, ii, :], axis=0) for ii in range(N_augments)])

# approximate answer
covs_by_realization = [np.cov(h, rowvar=False) for h in all_hists]

# correct answer
covs_by_augmentation = np.array([np.cov(all_hists[:, ii, :], rowvar=False) for ii in range(N_augments)])

fig, ax = plt.subplots(nrows=6, ncols=2,
                       gridspec_kw={'height_ratios': [1, 1, 1, 1, 2, 2]},
                       figsize=(15,23))

ax_stds = [ax[0,0], ax[0,1]]
ax_hists = [ax[1,0], ax[1,1]]
ax_vsfs = [ax[2,0], ax[2,1]]
ax_res = [ax[3,0], ax[3,1]]
ax_corrs = [ax[4,0], ax[4,1]]

gs = ax[5,0].get_gridspec()
for a in ax[5, :] :
    a.remove()
ax_allcorr = fig.add_subplot(gs[5, :])

def remove_frame(axis) :
    for s in ['top', 'bottom', 'right', 'left'] :
        axis.spines[s].set_visible(False)
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])

for a_s, a_h, a_v, a_r, a_c, s, v, c in \
    zip(ax_stds,
        ax_hists,
        ax_vsfs,
        ax_res,
        ax_corrs,
        [stds_by_realization,
         stds_by_augmentation],
        [vsfs_by_realization,
         vsfs_by_augmentation],
        [covs_by_realization,
         covs_by_augmentation]
       ):
    a_s.set_ylabel('std/sqrt(mean)')
    a_s.axhline(1, color='grey', linestyle='dashed')
    for s_ in s :
        a_s.plot(s_/np.sqrt(mean), linestyle='none', marker='o', markersize=0.4)
    a_s.set_ylim(0.6,1.4)
    to_hist = (s.reshape(-1, s.shape[-1]) / np.sqrt(mean)[None, :]).flatten()
    a_h.hist(to_hist, bins=np.linspace(0.7, 1.3, num=33))
    for v_ in v :
        a_v.plot(v_)
    a_v.errorbar(np.arange(0, v.shape[-1]), cmass_vsf, yerr=np.sqrt(mean), linestyle='none', marker='o', color='black', label='CMASS')
    a_v.plot(mean, linestyle='none', marker='x', color='black', label='fiducial mean')
    a_v.legend(loc='upper right')
    for v_ in v :
        a_r.plot(v_/mean - 1)
    a_r.set_ylim(-0.8, 1.0)

    corrs = [c_/np.sqrt(np.diagonal(c_)[:,None]*np.diagonal(c_)[None,:]) for c_ in c]
    nrows = 2
    ncols = 2
    pad_pix = 4
    indices = rng.choice(len(corrs), size=nrows*ncols, replace=False)
    corrs = [corrs[idx] for idx in indices]

    corr_rows, corr_cols = corrs[0].shape
    img = np.full((nrows*corr_rows+(nrows-1)*pad_pix,
                   ncols*corr_cols+(nrows-1)*pad_pix), float('nan'))
    for ii in range(nrows) :
        for jj in range(ncols) :
            img[ii*(corr_rows+pad_pix):ii*(corr_rows+pad_pix)+corr_rows,
                jj*(corr_cols+pad_pix):jj*(corr_cols+pad_pix)+corr_cols] = corrs[ii*ncols+jj]
    a_c.matshow(img, vmin=-1, vmax=1, cmap='seismic')
    a_c.set_ylabel('randomly picked correlation matrices')
    remove_frame(a_c)


    a_s.set_xlabel('VSF data vector index')
    a_v.set_xlabel('VSF data vector index')
    a_r.set_xlabel('VSF data vector index')
    a_h.set_xlabel('std/sqrt(mean)')
    a_s.set_ylabel('std/sqrt(mean)')
    a_v.set_ylabel('VSF')
    a_r.set_ylabel('VSF/<VSF> - 1')
    a_h.set_ylabel('counts')

    for a in [a_s, a_v, a_r, ] :
        a.set_xlim(-1, v.shape[-1])

# the big correlation matrix
allcov = np.cov(all_hists.reshape(-1, all_hists.shape[-1]), rowvar=False)
allcorr = allcov/np.sqrt(np.diagonal(allcov)[:,None]*np.diagonal(allcov)[None,:])
with np.printoptions(precision=2, suppress=True, threshold=sys.maxsize, linewidth=200) :
    print(allcorr)
im = ax_allcorr.matshow(allcorr, vmin=-1, vmax=1, cmap='seismic')
divider = make_axes_locatable(ax_allcorr)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax_allcorr.set_title('overall correlation matrix')
remove_frame(ax_allcorr)

ax[0,0].set_title('grouped by realization (quasi-independent augmentations)')
ax[0,1].set_title('grouped by augmentation (genuinely independent realizations)')


fig.savefig(f'check_vsf_likelihood_{NBINS}bins.pdf', bbox_inches='tight')
