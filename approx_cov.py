""" Compute approximate P_0(k) covariance matrix for our binning

This is not meant for production but rather to get some intuition
and find a good fiducial point
"""

import numpy as np
from scipy.interpolate import interp1d

patchy_k = np.loadtxt('/tigress/lthiele/boss_dr12/linear/Power_Spectrum_cmass_ngc_v5_Patchy_0001.txt',
                      usecols=0)
patchy_Nk = len(patchy_k)
patchy_cov = np.load('cov_cmass_ngc.npy')[:patchy_Nk,:patchy_Nk]
patchy_avg = np.load('avg_cmass_ngc.npy')[:patchy_Nk]

with np.load('boss_plk.npz') as fp :
    our_k = fp['k']
    our_boss_p0k = fp['p0k']
our_Nk = len(our_k)

patchy_sigma = np.sqrt(np.diag(patchy_cov))

patchy_ratio = patchy_sigma / patchy_avg
i = interp1d(patchy_k, patchy_ratio, fill_value='extrapolate')

# sigma ~ 1/sqrt(num_modes)
# and num_modes ~ k^2.
# Thus dN = (k+dk)^2 - k^2 = 2 dk/k
# and the ratio dN'/dN just scales as dk
dk_ratio = (patchy_k[1]-patchy_k[0]) / (our_k[1]-our_k[0])
approx_sigma = i(our_k) * our_boss_p0k * np.sqrt(dk_ratio)

approx_corr = np.eye(our_Nk)
# neighbouring bins, done by eye from Patchy covariance
approx_corr += 0.5 * np.eye(our_Nk, k=1)
approx_corr += 0.5 * np.eye(our_Nk, k=-1)
# next-to-neighbouring bins, again done by eye
approx_corr += 0.25 * np.eye(our_Nk, k=2)
approx_corr += 0.25 * np.eye(our_Nk, k=-2)

approx_cov = approx_corr * (approx_sigma[:,None] * approx_sigma[None,:])

np.save('approx_cov.npy', approx_cov)
