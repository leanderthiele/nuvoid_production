""" Exploration. Take optuna trials, fit a surrogate, and run MCMC on that.
Command line arguments:
    [1] optuna study name
"""

from sys import argv

import numpy as np

from scipy.spatial import Delaunay, ConvexHull
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import optuna
import emcee

NWALKERS = 128

study_name = argv[1]

study = optuna.load_study(study_name=study_name,
                          storage='mysql://optunausr:pwd@tigercpu:3310/optunadb'\
                                  '?unix_socket=/home/lthiele/mysql/mysql.sock')

trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,), deepcopy=False)

# parameter values
x = np.array([list(t.params.values()) for t in trials])
NDIM = x.shape[-1]
print(x.shape)

# objective values
y = np.array([t.value for t in trials])
print(y.shape)

# construct the Gaussian process
print('starting constructing gpr')
gpr = GPR()
gpr.fit(x, y)
print('finished constructing gpr')

# construct our "prior"
# print('starting constructing hull')
# hull = ConvexHull(x)
# print('finished constructing hull')

# construct initial positions for the walkers
sorted_trials = sorted(trials, key=lambda t: t.value)
theta_init = np.array([np.array(list(sorted_trials[ii].params.values()))
                       for ii in range(NWALKERS)])

def logprior(theta) :
    #if hull.find_simplex(theta) < 0 :
    #    return -np.inf
    for t, x_ in zip(theta, x.T) :
        if t < np.min(x_) or t > np.max(x_) :
            return -np.inf
    return 0.0

def loglikelihood(theta) :
    return -gpr.predict(theta.reshape(1,-1))

def logprob(theta) :
    lp = logprior(theta)
    if not np.isfinite(lp) :
        return -np.inf
    ll = loglikelihood(theta)
    return ll + lp

if __name__ == '__main__' :

    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, logprob)

    sampler.run_mcmc(theta_init, 1000, progress=True)

    chain = sampler.get_chain()
    np.save(f'test_mcmc_{study_name}.npy', chain)

    autocorr_times = sampler.get_autocorr_time()
    acceptance_rates = sampler.acceptance_fraction

    print(f'autocorr={autocorr_times}\n, acceptance={acceptance_rates}')

