from sys import argv
from glob import glob
import re
import multiprocessing as mp

import numpy as np

from scipy.optimize import minimize_scalar, root_scalar, dual_annealing, basinhopping
from sklearn.neighbors import KernelDensity

from lfi_load_posterior import load_posterior

outdir_base = argv[1]
num_workers = int(argv[2])

filebase = '/tigress/lthiele/nuvoid_production'
discard = 100 # burn in
# NOTE might have to trim chain, otherwise too slow
batch_size = num_workers * 10 # to get quicker results and prevent loss of data

match = re.search('coverage_chains_v(\d*)_([a-f,0-9]{32})_([a-f,0-9]{32})', outdir_base)
version = int(match[1])
compression_hash = match[2]
model_ident = match[3]

chain_fnames = glob(f'{filebase}/{outdir_base}/chain_*.npz')
with np.load(chain_fnames[0]) as f :
    param_names = list(f['param_names'])

settings, _ = load_posterior(f'{filebase}/lfi_model_v{version}_{compression_hash}_{model_ident}.sbi', None,
                             need_posterior=False)
priors = [settings['priors'][param_name] for param_name in param_names]

def oneminusalpha (samples, theta, prior) :

    std = np.std(samples)
    
    # approximate continuous posterior
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.1*std).fit(samples.reshape(-1, 1))
    nll = lambda x : -kde.score_samples(np.array([[x]]).reshape(-1,1))

    # get some feel for the posterior
    edges = np.linspace(*prior, num=501)
    centers = 0.5 * (bins[1:] + bins[:-1])
    h, _ = np.histogram(samples, bins=edges)
    x0 = centers[np.argmax(h)]
    xlo = max((prior[0], x0-std))
    xhi = min((prior[1], x0+std))
    while not np.isfinite(nll(xlo)) :
        xlo += 1e-2 * std
    while not np.isfinite(nll(xhi)) :
        xhi -= 1e-2 * std

    # find maximum posterior so we know in which direction to go
    try :
        sln = basinhopping(nll, x0, T=0.1, niter=10, minimizer_kwargs={'bounds': [(xlo, xhi), ])
    except ValueError : # rare case
        print(f'*** basinhopping failed')
        return -1

    xmax = sln.x
    side = 'l' if theta<xmax else 'r'
    ytarg = nll(theta)
    edge = prior[1 if side=='l' else 0]
    if nll(edge) < ytarg :
        # pathological one-sided case
        xsln = edge
    else :
        ftarg = lambda x : ytarg - nll(x)
        bounds = (xmax, np.max(samples)) if side=='l' else (np.min(samples), xmax)
        try :
            sln = root_scalar(ftarg, bracket=bounds)
            xsln = sln.x
        except Exception as e :
            if str(e) == 'f(a) and f(b) must have different signs' :
                # extremely rare case in which theta is exactly at the maximum posterior
                return 0.0
            print(e)
            return -1

    xmin, xmax = (theta, xsln) if side=='l' else (xsln, theta)
    return np.sum((samples>xmin)*(samples<xmax)) / len(samples)

def job (chain_fname) :

    match = re.search('chain_(\d*)_cosmo(\d*).npz', chain_fname)
    obs_idx  = int(match[1])
    cosmo_idx = int(match[2])

    with np.load(chain_fname) as f :
        param_names_ = list(f['param_names'])
        chain = f['chain']
        real_params = f['real_params']

    assert param_names == param_names_
    chain = chain[discard:].reshape(-1, chain.shape[-1])

    oma = [oneminusalpha(x, theta, prior) for x, theta, prior \
           in zip(chain.T, real_params, priors)]

    print(f'obs_idx={obs_idx} done!')
    return obs_idx, cosmo_idx, oma


if __name__ == '__main__' :
    
    batch_idx = 0
    while True : # keep looping over batches
        
        these_chain_fnames = chain_fnames[batch_idx*batch_size : (batch_idx+1)*batch_size]
        with mp.Pool(num_workers) as pool :
            results = pool.map(job, these_chain_fnames)

        obs_idx = [r[0] for r in results]
        cosmo_idx = [r[1] for r in results]
        oma=[r[2] for r in results]

        np.savez(f'{filebase}/oneminusalpha_v{version}_{compression_hash}_{model_ident}_batch{batch_idx}.npz',
                 obs_idx=np.array(obs_idx),
                 cosmo_idx=np.array(cosmo_idx),
                 oneminusalpha=np.array(oma))
