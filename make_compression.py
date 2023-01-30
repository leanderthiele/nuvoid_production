from sys import argv

import numpy as np

from cut_compress import CutCompress
from derivatives import LinRegress

version = int(argv[1])

consider_params = [
#                   'Obh2',
#                   'Och2',
#                   'theta',
#                   'logA',
#                   'ns',
                   'Mnu',
#                   'hod_transfP1',
#                   'hod_abias',
                   'hod_log_Mmin',
                   'hod_sigma_logM',
#                   'hod_log_M0',
#                   'hod_log_M1',
                   'hod_alpha',
#                   'hod_transf_eta_cen',
#                   'hod_transf_eta_sat',
                   'hod_mu_Mmin',
#                   'hod_mu_M1',
                  ]
is_nuisance = np.array([not (x in consider_params) for x in LinRegress.use_params], dtype=bool)

codebase = '/home/lthiele/nuvoid_production'

cmb_prior = np.loadtxt(f'{codebase}/mu_cov_plikHM_TTTEEE_lowl_lowE.dat', skiprows=3)

mnu_sigma = 0.5 # some reasonable number to approximate prior

# approximate effect of prior boundaries here to avoid pathologies
hod_sigmas = [
              3.0, # transfP1
              1.0, # abias
              0.4, # log_Mmin
              0.4, # sigma_logM
              1.5, # log_M0
              1.5, # log_M1
              0.7, # alpha
              2.5, # transf_eta_cen
              1.0, # transf_eta_sat
              20.0, # mu_Mmin
              40.0, # mu_M1
             ]

Cprior = np.zeros((17, 17))
Cprior[:5, :5] = cmb_prior
Cprior[5, 5] = mnu_sigma**2
Cprior[6:, 6:] = np.diagflat(np.array(hod_sigmas)**2)

linregress = LinRegress(version, cut=None)
compress = CutCompress(linregress.dm_dphi, linregress.cov, is_nuisance, Cprior=Cprior,
                       use_vsf=True, use_vgplk=True, use_plk=True,
                       vsf_zbins=[0,1], vsf_Rmin=30, vsf_Rmax=80,
                       vgplk_Rbins=[30, 40, 50], vgplk_ell=[0, 2],
                       plk_ell=[0, 2],
                       kmin=0.02, kmax=0.1)

np.set_printoptions(formatter={'all': lambda x: '%+.2e'%x}, linewidth=200, threshold=1000000)

F = compress.compress.F_phi
C = np.linalg.inv(F)
print(f'parameters={consider_params}')
print(f'Fisher approximation covariance matrix =\n{C}')

np.savez('compression.npz',
         norm=linregress.fid_mean,
         m=compress.cut_compression_matrix,
         fisher_covmat=C)
