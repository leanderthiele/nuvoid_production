from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RMIN = 40

if torch.cuda.is_available() :
    device = 'cuda'
else :
    device = 'cpu'

torch.manual_seed(42)
rng = np.random.default_rng(42)

VALIDATION_FRAC = 0.2

with np.load('/tigress/lthiele/collected_vgplk.npz') as f :
    cosmo_indices = f['cosmo_indices']
    params = f['params']
    Rmin = f['Rmin']
    k = f['k']
    vgplk = f['vgplk'] # shape [params, 8, Rmin, ell, k]
    Nvoids = f['Nvoids'] # shape [params, 8, Rmin]

with np.load('/tigress/lthiele/cov_vgplk.npz') as f :
    cov = f['cov'] # shape [Rmin, ellxk]
    assert np.allclose(Rmin, f['Rmin'])
    k_indices = f['k_indices']

Nvoids_fid_all = {30: 1463.39, 40: 786.59, 50: 315.81}
Nvoids_fid = Nvoids_fid_all[RMIN]

# choose Rmin
rmin_idx = np.where(Rmin == RMIN)
vgplk = vgplk[:, :, rmin_idx, :, :]
Nvoids = Nvoids[:, :, rmin_idx]
cov = cov[rmin_idx, ...]

# choose k and flatten ellxk
vgplk = vgplk[..., k_indices]
vgplk = vgplk.reshape(*vgplk.shape[:2], -1) # shape [params, 8, ellxk]


# split by cosmologies
uniq_cosmo_indices = np.unique(cosmo_indices)
validation_cosmo_indices = rng.choice(uniq_cosmo_indices, replace=False,
                                      size=int(VALIDATION_FRAC * len(uniq_cosmo_indices)))
validation_select = np.array([idx in validation_cosmo_indices for idx in cosmo_indices], dtype=bool)

train_x = params[~validation_select]
validation_x = params[validation_select]
train_vgplk = vgplk[~validation_select]
validation_vgplk = vgplk[validation_select]
train_Nvoids = Nvoids[~validation_select]
validation_Nvoids = Nvoids[validation_select]

# normalize inputs
avg = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x = (train_x - avg[None, :]) / std[None, :]
validation_x = (validation_x - avg[None, :]) / std[None, :]
np.savez('vgplk_norm.npz', avg=avg, std=std)

# repeat the parameters, flatten vgplk

# shapes [paramsx8, Nparams]
train_x = train_x.repeat(8, axis=0)
validation_x = validation_x.repeat(8, axis=0)

# shapes [paramsx8, ellxk]
train_vgplk = train_vgplk.reshape(-1, train_vgplk.shape[-1])
validation_vgplk = validation_vgplk.reshape(-1, validation_vgplk.shape[-1])

# choose the runs where we have data
select = np.all(np.isfinite(train_vgplk), axis=-1)
train_x = train_x[select]
train_vgplk = train_vgplk[select]
train_Nvoids = train_Nvoids[select]
assert np.all(np.isfinite(train_vgplk))
assert np.all(np.isfinite(train_Nvoids))

select = np.all(np.isfinite(validation_vgplk), axis=-1)
validation_x = validation_x[select]
validation_vgplk = validation_vgplk[select]
validation_Nvoids = validation_Nvoids[select]
assert np.all(np.isfinite(validation_vgplk))
assert np.all(np.isfinite(validation_Nvoids))

# precision matrix
covinv = np.linalg.inv(cov)

# to torch
train_x = torch.from_numpy(train_x.astype(np.float32)).to(device=device)
train_vgplk = torch.from_numpy(train_vgplk.astype(np.float32)).to(device=device)
train_Nvoids = torch.from_numpy(train_Nvoids.astype(np.float32)).to(device=device)
validation_x = torch.from_numpy(validation_x.astype(np.float32)).to(device=device)
validation_vgplk = torch.from_numpy(validation_vgplk.astype(np.float32)).to(device=device)
validation_Nvoids = torch.from_numpy(validation_Nvoids.astype(np.float32)).to(device=device)

class MLPLayer(nn.Sequential) :
    def __init__(self, Nin, Nout, activation=nn.LeakyReLU) :
        super().__init__(OrderedDict([('linear', nn.Linear(Nin, Nout, bias=True)),
                                      ('activation', activation()),
                                     ]))

class MLP(nn.Sequential) :
    def __init__(self, Nin, Nout, Nlayers=4, Nhidden=512, out_positive=True) :
        # output is manifestly positive so we use ReLU in the final layer
        self.Nin = Nin
        self.Nout = Nout
        super().__init__(*[MLPLayer(Nin if ii==0 else Nhidden,
                                    Nout if ii==Nlayers else Nhidden,
                                    activation=(nn.LeakyReLU if (ii!=Nlayers or not out_positive) else nn.ReLU))
                           for ii in range(Nlayers+1)])

class Loss(nn.Module) :
    def __init__(self) :
        super().__init__()
    def forward(self, pred, targ, n) :
        # shapes are pred, targ: [batch, data], n: [batch]
        delta = pred - targ
        x = torch.einsum('...i,ij,...j->...', delta, covinv, delta)
        scaling = n / Nvoids_fid
        return torch.mean(scaling * x)

EPOCHS = 100

train_set = TensorDataset(train_x, train_vgplk, train_Nvoids)
validation_set = TensorDataset(validation_x, validation_vgplk, validation_Nvoids)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=512)

loss = Loss()
model = MLP(train_x.shape[-1], train_vgplk.shape[-1], out_positive=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=EPOCHS, verbose=True)

for epoch in range(EPOCHS) :
    
    model.train()
    ltrain = []
    for x, y, n in train_loader :
        optimizer.zero_grad()
        pred = model(x)
        l = loss(pred, y, n)
        ltrain.append(l.item())
        l.backward()
        optimizer.step()
    scheduler.step()
    ltrain = np.mean(np.array(ltrain))

    model.eval()
    lvalidation = []
    for x, y, n in validation_loader :
        pred = model(x)
        l = loss(pred, y, n)
        lvalidation.append(l.item())
    lvalidation = np.mean(np.array(lvalidation))

    print(f'iteration {epoch:4}: {ltrain:16.2f}\t{lvalidation:16.2f}')
