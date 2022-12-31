import numpy as np

KMIN = 0.025
KMAX = 0.1

f = np.load('fiducial_vgplk.npz')
k = f['k']
Rmin = f['Rmin']
seeds = list(map(lambda s: int(s.split('_')[1]), list(filter(lambda s: 'seed' in s, list(f.keys())))))

k_indices = np.where((k>KMIN) * (k<KMAX))[0]
print(k_indices)

all_vgplk = np.stack([f[f'seed_{seed}'] for seed in seeds], axis=0)
all_vgplk = all_vgplk[..., k_indices]

covs = []
for rmin_idx in range(len(Rmin)) :
    x = all_vgplk[:, :, rmin_idx, :]
    x = x.reshape(*x.shape[:2], -1)
    x = x.reshape(-1, x.shape[-1])
    select = np.all(np.isfinite(x), axis=-1)
    x = x[select]
    c = np.cov(x, rowvar=False)
    covs.append(c)

np.savez('cov_vgplk.npz', cov=np.array(covs), k_indices=k_indices, Rmin=Rmin)
