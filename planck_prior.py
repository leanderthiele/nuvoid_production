import numpy as np

PARAMS = ['omegabh2', 'omegach2', 'theta', 'logA', 'ns', ]
FACTOR = 2 # by which we enlarge the error bars

LIKE = 'plikHM_TTTEEE_lowl_lowE'
LENSING = True
OTHER = ''

like_ = LIKE.replace('_', '-')
base = f'/home/lthiele/COM_CosmoParams_base-{like_}_R3.00'

directory = LIKE + ('_lensing' if LENSING else '')
ident = directory + OTHER

fname_covmat = f'{base}/base/{directory}/dist/base_{ident}.covmat'
fname_margestats = f'{base}/base/{directory}/dist/base_{ident}.margestats'
fname_out = f'./mu_cov_{ident}.dat'

# find the indices corresponding to our parameters in the covariance matrix
with open(fname_covmat, 'r') as f :
    header = f.readline().strip().split()
indices = [header.index(p)-1 for p in PARAMS] # subtract 1 to get rid of '#'

cov_all = np.loadtxt(fname_covmat)
cov = cov_all[indices,:][:,indices]

cov *= FACTOR**2

# find the mean values
avg = [None, ] * len(PARAMS)
with open(fname_margestats, 'r') as f :
    for line in f :
        line = line.split()
        if len(line) == 0 :
            continue
        for ii, p in enumerate(PARAMS) :
            if line[0] == p :
                assert avg[ii] is None
                avg[ii] = float(line[1])
avg = np.array(avg)

with open(fname_out, 'w') as f :
    f.write('# %s\n'%(', '.join(PARAMS)))
    f.write(f'# mean {fname_margestats}:\n')
    np.savetxt(f, avg[None, :])
    f.write(f'# covariance {fname_covmat}:\n')
    np.savetxt(f, cov)
