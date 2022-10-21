import numpy as np

PARAMS = ['omegabh2', 'omegach2', 'theta', 'tau', 'logA', 'ns', ]

LIKE = 'plikHM-TTTEEE-lowl-lowE'
BASE = f'/home/lthiele/COM_CosmoParams_base-{LIKE}_R3.00'
LENSING = True
OTHER = ''

directory = LIKE + ('_lensing' if LENSING else '')
ident = directory + OTHER

fname_covmat = f'{BASE}/base/{directory}/dist/{ident}.covmat'
fname_margestats = f'{BASE}/base/{directory}/dist/{ident}.margestats'

# find the indices corresponding to our parameters in the covariance matrix
with open(fname_covmat, 'r') as f :
    header = f.readline().strip().split()
indices = [header.index(p)-1 for p in PARAMS] # subtract 1 to get rid of '#'

cov_all = np.loadtxt(fname_covmat)
cov = cov_all[indices,:][:,indices]

# find the mean values
avg = [None, ] * len(PARAMS)
with open(fname_margestats, 'r') as f :
    for line in f :
        for ii, p in enumerate(PARAMS) :
            if line.startswith(p) :
                assert avg[ii] is None
                avg[ii] = float(line.split()[1])

print(cov)
print(avg)
