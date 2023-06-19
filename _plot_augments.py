import numpy as np
from scipy.stats import wishart as wishart_gen
from matplotlib import pyplot as plt

from read_txt import read_txt
from _plot_datavec import plot_datavec, xindices, figsize
from _plot_style import *

version = 0
compression_hash = 'faae54307696ccaff07aef77d20e1c1f'

filebase = '/tigress/lthiele/nuvoid_production'
fiducials_fname = f'{filebase}/datavectors_fiducials_v{version}.npz'
compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'

with np.load(fiducials_fname) as f :
    data = f['data']
    sim_idx = f['sim_idx']
    lc_idx = f['lc_idx']

compress_norm = read_txt(compress_fname, 'normalization:')
compress_matrix = read_txt(compress_fname, 'compression matrix:')

# first pass to get best covariances
data = data / compress_norm[None, :]
cmp_data = np.einsum('ab,ib->ia', compress_matrix, data)

cov = np.cov(data, rowvar=False)
cmp_cov = np.cov(cmp_data, rowvar=False)

uniq_sim_idx = np.unique(sim_idx)
uniq_lc_idx = np.unique(lc_idx)
num_avail = np.array([np.count_nonzero(lc_idx==ii) for ii in uniq_lc_idx])
uniq_lc_idx = uniq_lc_idx[num_avail == len(uniq_sim_idx)]

# downsample to equal sims and lightcones
rng = np.random.default_rng(137)
uniq_lc_idx = rng.choice(uniq_lc_idx, len(uniq_sim_idx), replace=False)
assert len(uniq_sim_idx) == len(uniq_lc_idx)

# choose only the data that have the lightcone index we need
mask = np.array([ii in uniq_lc_idx for ii in lc_idx])
data = data[mask]
cmp_data = cmp_data[mask]
sim_idx = sim_idx[mask]
lc_idx = lc_idx[mask]

def do_job(x, all_cov, is_compressed) :

    covs_by_sim = [np.cov(x[sim_idx==ii], rowvar=False) for ii in uniq_sim_idx]
    covs_by_lc = [np.cov(x[lc_idx==ii], rowvar=False) for ii in uniq_lc_idx]

    if is_compressed :
        fig_sigma, ax_sigma = plt.subplots(nrows=1, ncols=1, figsize=(5,1.5))
    else :
        fig_sigma, _ax_sigma = plot_datavec(color='grey', alpha=0.3)
        ax_sigma = _ax_sigma.twinx()
    sigmas_by_sim = [np.sqrt(np.diagonal(c)) for c in covs_by_sim]
    sigmas_by_lc = [np.sqrt(np.diagonal(c)) for c in covs_by_lc]
    avg_sigmas_by_sim = np.mean(np.array(sigmas_by_sim), axis=0)
    avg_sigmas_by_lc = np.mean(np.array(sigmas_by_lc), axis=0)
    xi = np.arange(x.shape[1]) if is_compressed else xindices
    y = avg_sigmas_by_sim/avg_sigmas_by_lc
    ax_sigma.plot(xi, y,
                  linestyle='none', marker='x')
    ydelta = np.max(np.fabs(y-1)) + 0.1*np.std(y)
    ax_sigma.set_ylim(1-ydelta, 1+ydelta)
    ax_sigma.set_ylabel('$\sigma({\sf augments})/\sigma({\sf ICs})$')
    ax_sigma.yaxis.tick_left()
    ax_sigma.yaxis.set_label_position('left')
    if is_compressed :
        ax_sigma.set_xlabel('compressed data vector index')
        ax_sigma.axhline(1, color='grey', linestyle='dashed')

    if is_compressed :
        # wishart distribution parameters
        n = len(uniq_lc_idx)
        p = len(all_cov)
        scale = all_cov / n
        scale_inv = np.linalg.inv(scale)

        def wishart(C) :
            return -(n-p-1)*np.log(np.linalg.det(C)) + np.trace(scale_inv @ C)

        wishart_by_sim = np.array([wishart(c) for c in covs_by_sim])
        wishart_by_lc = np.array([wishart(c) for c in covs_by_lc])
        edges = np.histogram_bin_edges(np.concatenate([wishart_by_sim, wishart_by_lc]), bins=10)

        gen = wishart_gen(df=n, scale=scale)
        wishart_rvs = np.array([wishart(gen.rvs()) for _ in range(1000)])

        fig_wishart, ax_wishart = plt.subplots(nrows=1, ncols=1, figsize=(5,1.5))
        hist_kwargs = dict(histtype='step')
        ax_wishart.hist(wishart_by_sim, bins=edges, label='augments', **hist_kwargs)
        ax_wishart.hist(wishart_by_lc, bins=edges, label='ICs', **hist_kwargs)
        y, e = np.histogram(wishart_rvs, bins=edges)
        y = y.astype(float) / np.sum(y) * len(uniq_lc_idx)
        c = 0.5 * (e[1:] + e[:-1])
        ax_wishart.plot(c, y, label='expected')
        ax_wishart.set_xlabel('$-2\log p_{\sf Wishart}$')
        ax_wishart.set_ylabel('counts')
        ax_wishart.legend(loc='upper right', frameon=False)
    else :
        fig_wishart, ax_wishart = None, None

    return fig_sigma, fig_wishart

for x, C, is_cmp, ident in zip([data, cmp_data, ],
                               [cov, cmp_cov, ],
                               [False, True, ],
                               ['raw', 'compressed', ]) :

    fs, fw = do_job(x, C, is_cmp)

    for f, stat in zip([fs, fw, ], ['sigma', 'wishart', ]) :
        if f is not None :
            savefig(f, f'augments_{stat}_{ident}')
