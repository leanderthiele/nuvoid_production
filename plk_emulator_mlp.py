from sys import argv
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available() :
    device = 'cuda'
else :
    device = 'cpu'

torch.manual_seed(42)
rng = np.random.default_rng(42)

VALIDATION_FRAC = 0.2

KMIN = 0.025

with np.load('/tigress/lthiele/plk_grouped_data.npz') as fp :
    cosmo_indices = fp['cosmo_indices']
    params = fp['params']
    p0k_samples = f['p0k']
with np.load('/tigress/lthiele/boss_dr12/plk/boss_plk.npz') as fp :
    k = fp['k']
approx_cov = np.load('/tigress/lthiele/boss_dr12/plk/approx_cov.npy')

# low k are weird, so we exclude them to avoid instability
min_idx = np.argmin(np.fabs(k-KMIN))
p0k_samples = p0k_samples[..., min_idx:]
k = k[min_idx:]
approx_cov = approx_cov[min_idx:, min_idx:]

# split by cosmologies to have as much independence as possible in test set
uniq_cosmo_indices = np.unique(cosmo_indices)
validation_cosmo_indices = rng.choice(uniq_cosmo_indices, replace=False,
                                      size=int(VALIDATION_FRAC * len(uniq_cosmo_indices)))
validation_select = np.array([idx in validation_cosmo_indices for idx in cosmo_indices], dtype=bool)

train_x = params[~validation_select]
validation_x = params[validation_select]

# normalize
avg = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x = (train_x - avg[None, :]) / std[None, :]
validation_x = (validation_x - avg[None, :]) / std[None, :]
np.savez('plk_norm.npz', avg=avg, std=std)

# first try: only train on the averages, this should still be enough samples hopefully
p0k_target = np.mean(p0k_samples, axis=1)
approx_cov /= p0k_samples.shape[1]
train_p0k = p0k_target[~validation_select]
validation_p0k = p0k_target[validation_select]

# precision matrix
covinv = np.linalg.inv(approx_cov)

# now to torch
train_x = torch.from_numpy(train_x.astype(np.float32)).to(device=device)
validation_x = torch.from_numpy(validation_x.astype(np.float32)).to(device=device)
train_p0k = torch.from_numpy(train_p0k.astype(np.float32)).to(device=device)
validation_p0k = torch.from_numpy(validation_p0k.astype(np.float32)).to(device=device)
covinv = torch.from_numpy(covinv.astype(np.float32)).to(device=device)

class MLPLayer(nn.Sequential) :
    def __init__(self, Nin, Nout, activation=nn.LeakyReLU) :
        super().__init__(OrderedDict([('linear', nn.Linear(Nin, Nout, bias=True)),
                                      ('activation', activation()),
                                     ]))

class MLP(nn.Sequential) :
    def __init__(self, Nin, Nout, Nlayers=8, Nhidden=512, out_positive=True) :
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
    def forward(self, x1, x2) :
        # shapes are [..., Nk]
        delta = x1-x2
        x = torch.einsum('...,i,ij,...j->...', delta, covinv, delta) # this has still the leading shapes
        return torch.mean(x)

EPOCHS = 100

train_set = TensorDataset(train_x, train_p0k)
validation_set = TensorDataset(validation_x, validation_p0k)
train_loader = DataLoader(train_set, batch_size=512)
validation_loader = DataLoader(validation_set, batch_size=512)
loss = Loss()
model = MLP(train_x.shape[-1], train_p0k.shape[-1], out_positive=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS) :

    model.train()
    ltrain = []
    for x, y in train_loader :
        optimizer.zero_grad()
        pred = model(x)
        l = loss(pred, y)
        ltrain.append(l.item())
        l.backward()
        optimizer.step()
    ltrain = np.mean(np.array(ltrain))

    model.eval()
    lvalidation = []
    for x, y in validation_loader :
        pred = model(x)
        l = loss(pred, y)
        lvalidation.append(l.item())
    lvalidation = np.mean(np.array(lvalidation))

    print(f'iteration {epoch:4}: {ltrain:8.2f}\t{lvalidation:8.2f}')
