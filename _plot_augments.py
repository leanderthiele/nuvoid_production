from sys import argv

import numpy as np
from scipy.stats import wishart as wishart_gen
from matplotlib import pyplot as plt

from read_txt import read_txt
from _plot_datavec import plot_datavec, xindices, figsize

plt.style.use('dark_background')

version = 0
compression_hash = argv[1]

filebase = '/tigress/lthiele/nuvoid_production'
fiducials_fname = f'{filebase}/datavectors_fiducials_v{version}.npz'
compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.npz'

with np.load(fiducials_fname) as f :
    data = f['data']
    sim_idx = f['sim_idx']
    lc_idx = f['lc_idx']
uniq_sim_idx = np.unique(sim_idx)
uniq_lc_idx = np.unique(lc_idx)

# downsample to equal sims and lightcones
rng = np.random.default_rng(137)
uniq_lc_idx = rng.choice(uniq_lc_idx, len(uniq_sim_idx), replace=False)

assert len(uniq_sim_idx) == len(uniq_lc_idx)

compress_norm = read_txt(compress_fname, 'normalization:')
compress_matrix = read_txt(compress_fname, 'compression matrix:')

data = data / compress_norm[None, :]
cmp_data = np.einsum('ab,ib->ia', compress_matrix, data)

def do_job(x, is_compressed) :

    all_cov = np.cov(x, rowvar=False)
    covs_by_sim = [np.cov(x[sim_idx==ii], rowvar=False) for ii in uniq_sim_idx]
    covs_by_lc = [np.cov(x[lc_idx=ii] for ii in uniq_lc_idx]

    if is_compressed :
        fig_sigma, ax_sigma = plt.subplots(nrows=1, ncols=1, figsize=(5,3))
    else :
        fig_sigma, _ax_sigma = plot_datavec()
        ax_sigma = _ax_sigma.twinx()
    sigmas_by_sim = [np.sqrt(np.diagonal(c) for c in covs_by_sim]
    sigmas_by_lc = [np.sqrt(np.diagonal(c) for c in covs_by_lc]
    avg_sigmas_by_sim = np.mean(np.array(sigmas_by_sim), axis=0)
    avg_sigmas_by_lc = np.mean(np.array(sigmas_by_lc), axis=0)
    ax_sigma.plot(xindices, avg_sigmas_by_sim/avg_sigmas_by_lc,
                  linestyle='none', marker='x')
    ax_sigma.set_ylabel('$\sigma({\sf augments})/\sigma({\sf ICs})$')
    if is_compressed :
        ax_sigma.set_xlabel('compressed data vector index')

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

        fig_wishart, ax_wishart = plt.subplots(nrows=1, ncols=1, figsize=(5,3))
        hist_kwargs = dict(histtype='step')
        ax_wishart.hist(wishart_by_sim, bins=edges, label='augments', **hist_kwargs)
        ax_wishart.hist(wishart_by_lc, bins=edges, label='ICs', **hist_kwargs)
        y, e = np.histogram(wishart_rvs, bins=edges)
        y = y.astype(float) / np.sum(y) * len(uniq_lc_idx)
        c = 0.5 * (e[1:] + e[:-1])
        ax_wishart.plot(c, y, label='expected')
        ax_wishart.set_xlabel('$-2\log p_{\sf Wishart}$')
        ax_wishart.set_ylabel('counts')
    else :
        fig_wishart, ax_wishart = None, None

    return fig_sigma, fig_wishart

for x, is_cmp, ident in zip([data, cmp_data, ],
                            [False, True, ],
                            ['raw', 'compressed', ]) :

    fs, fw = do_job(x, is_cmp)

    for f, stat in zip([fs, fw, ], [sigma, wishart, ]) :
        if f is not None :
            f.savefig(f'_plot_augments_{stat}_{ident}.png', bbox_inches='tight', transparent=False)
