import sys
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

RMIN = int(argv[1])
KMIN = 0.025
KMAX = 0.1

f = np.load('fiducial_vgplk.npz')
k = f['k']
Rmin = f['Rmin']
seeds = list(map(lambda s: int(s.split('_')[1]), list(filter(lambda s: 'seed' in s, list(f.keys())))))

min_idx = np.argmin(np.fabs(k-KMIN))
max_idx = np.argmin(np.fabs(k-KMAX))
rmin_idx = np.argmin(np.fabs(Rmin-RMIN))

with np.load('cmass_vgplk.npz') as f1 :
    assert np.allclose(f1['k'], k)
    cmass_vgplk = np.stack([f1[f'p{ell}k_Rmin{Rmin[rmin_idx]}'] for ell in [0, 2]], axis=0)
cmass_vgplk = cmass_vgplk[..., min_idx : max_idx+1].flatten()

# shape [seed, augment, Rmin, ell, k]
all_vgplk = np.stack([f[f'seed_{seed}'] for seed in seeds], axis=0)[..., min_idx : max_idx+1][:, :, rmin_idx, :]
all_vgplk = all_vgplk.reshape(*all_vgplk.shape[:2], -1)
print(all_vgplk.shape)

mean = np.nanmean(all_vgplk.reshape(-1, *all_vgplk.shape[2:]), axis=0)
print(mean.shape)

# downsample augmentations
N_augments = len(seeds)
rng = np.random.default_rng(42)
if True :
    # doesn't work at the moment
    good_augments = np.where(np.all(np.isfinite(all_vgplk.reshape(*all_vgplk.shape[:2], -1)[:, :, 0]), axis=0))[0]
    assert len(good_augments)>=N_augments, len(good_augments)
    use_augments = rng.choice(good_augments, N_augments, replace=False)
    all_vgplk = all_vgplk[:, use_augments, ...]
else :
    all_vgplk = np.stack([v[rng.choice(np.where(np.isfinite(v.reshape(v.shape[0], -1)[:, 0]))[0],
                                       N_augments, replace=False)] for v in all_vgplk],
                         axis=0)
assert np.all(np.isfinite(all_vgplk))

# the big correlation matrix
allcov = np.cov(all_vgplk.reshape(-1, all_vgplk.shape[-1]), rowvar=False)
allcorr = allcov/np.sqrt(np.diagonal(allcov)[:,None]*np.diagonal(allcov)[None,:])

# the stds have shape [realization/augmentation, Rmin, ell, k]

# this is the approximate answer
stds_by_realization = np.array([np.std(v, axis=0) for v in all_vgplk])

# this is the correct (reference answer)
stds_by_augmentation = np.array([np.std(all_vgplk[:, ii, ...], axis=0) for ii in range(N_augments)])

# for normalization
std_for_norm = np.std(all_vgplk.reshape(-1, *all_vgplk.shape[2:]), axis=0)

# approximate answer
vgplks_by_realization = np.array([np.mean(v, axis=0) for v in all_vgplk])

# reference answer
vgplks_by_augmentation = np.array([np.mean(all_vgplk[:, ii, ...], axis=0) for ii in range(N_augments)])

# approximate answer
covs_by_realization = [np.cov(v, rowvar=False) for v in all_vgplk]

# correct answer
covs_by_augmentation = np.array([np.cov(all_vgplk[:, ii, :], rowvar=False) for ii in range(N_augments)])

def cov_distances(l1, l2) :
    # returns an off-diagonal triangle of the matrix whose entries are the matrix norms
    # (we are using simple Frobenius) of the pairwise differences between the lists of
    # covariance matrices l1, l2
    assert len(l1) == len(l2)

    # we divide by this matrix to take out the overall variation in sigma
    norm_mat = np.sqrt(np.diagonal(allcov)[:,None]*np.diagonal(allcov)[None,:])
    out = []
    for ii in range(len(l1)) :
        for jj in range(ii) :
            delta = (l1[ii] - l2[jj]) / norm_mat
            out.append(np.linalg.norm(delta, ord='fro'))
    return np.array(out)


fig, ax = plt.subplots(nrows=5, ncols=2,
                       gridspec_kw={'height_ratios': [1, 1, 1, 2, 2]},
                       figsize=(15,25))
if len(ax.shape) == 1 :
    ax = np.array([ax,])

ax_stds = [ax[0,0], ax[0,1]]
ax_hists = [ax[1,0], ax[1,1]]
ax_vgplks = [ax[2,0], ax[2,1]]
ax_corrs = [ax[3,0], ax[3,1]]
ax_allcorr = ax[4,0]
ax_disthists = ax[4,1]

#gs = ax[4,0].get_gridspec()
#for a in ax[4, :] :
#    a.remove()
#ax_allcorr = fig.add_subplot(gs[4, :])

def remove_frame(axis) :
    for s in ['top', 'bottom', 'right', 'left'] :
        axis.spines[s].set_visible(False)
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])

for a_s, a_h, a_v, a_c, s, v, c in \
    zip(ax_stds,
        ax_hists,
        ax_vgplks,
        ax_corrs,
        [stds_by_realization,
         stds_by_augmentation],
        [vgplks_by_realization,
         vgplks_by_augmentation],
        [covs_by_realization,
         covs_by_augmentation]
       ):
    a_s.axhline(1, color='grey', linestyle='dashed')
    for s_ in s :
        a_s.plot(s_/std_for_norm, linestyle='none', marker='o', markersize=0.4)
    to_hist = (s.reshape(-1, s.shape[-1]) / std_for_norm[None, :]).flatten()
    a_h.hist(to_hist, bins=np.linspace(0.7, 1.3, num=33))
    for v_ in v :
        a_v.plot(v_)
    a_v.errorbar(np.arange(0, v.shape[-1]), cmass_vgplk, yerr=std_for_norm, linestyle='none', marker='o',
                 color='black', label='CMASS')
    a_v.plot(mean, color='black', linewidth=2, label='fiducial mean')
    a_v.legend(loc='lower right')

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

    a_s.set_xlabel('vgplk data vector index')
    a_h.set_xlabel('std/overall std')
    a_v.set_xlabel('vgplk data vector index')
    a_s.set_ylabel('std/overall std')
    a_h.set_ylabel('counts')
    a_v.set_ylabel('vgplk')

with np.printoptions(precision=2, suppress=True, threshold=sys.maxsize, linewidth=200) :
    print(allcorr)
im = ax_allcorr.matshow(allcorr, vmin=-1, vmax=1, cmap='seismic')
divider = make_axes_locatable(ax_allcorr)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax_allcorr.set_title('overall correlation matrix')
remove_frame(ax_allcorr)

# the covariance matrix distances
d_rr = cov_distances(covs_by_realization, covs_by_realization)
d_aa = cov_distances(covs_by_augmentation, covs_by_augmentation)
d_ra = cov_distances(covs_by_realization, covs_by_augmentation)
edges = np.histogram_bin_edges(np.concatenate([d_rr, d_aa, d_ra]), bins=30)
kwargs = dict(histtype='step', linewidth=2)
ax_disthists.hist(d_rr, bins=edges, label='r-r', **kwargs)
ax_disthists.hist(d_aa, bins=edges, label='a-a', **kwargs)
ax_disthists.hist(d_ra, bins=edges, label='r-a', **kwargs)
ax_disthists.legend()
ax_disthists.set_xlabel('covariance matrix distance')
ax_disthists.set_ylabel('counts')

ax[0,0].set_title('grouped by realization (quasi-independent augmentations)')
ax[0,1].set_title('grouped by augmentation (genuinely independent realizations)')

fig.savefig(f'check_vgplk_likelihood_Rmin{RMIN}.pdf', bbox_inches='tight')
