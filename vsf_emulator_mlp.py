# Command line arguments:
#   [1] file containing the training data

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

VALIDATION_FRAC = 0.2

data_file = argv[1]

with np.load(data_file) as f :
    param_names = list(f['param_names'])
    params = f['params']
    hists  = f['hists']

N = len(params)
assert N == hists.shape[0]

rng = np.random.default_rng(42)
validation_select = rng.choice([True, False], size=N, p=[VALIDATION_FRAC, 1.0-VALIDATION_FRAC])

train_params = params[~validation_select]
train_hists = hists[~validation_select]
validation_params = params[validation_select]
validation_hists = hists[validation_select]

# subsample training set to check convergence
p = None
if p is not None :
    select = rng.choice([True, False], size=len(train_params), p=[p, 1.0-p])
    train_params = train_params[select]
    train_hists = train_hists[select]

# normalize the inputs
avg = np.mean(train_params, axis=0)
std = np.std(train_params, axis=0)
train_params = (train_params - avg[None, :]) / std[None, :]
validation_params = (validation_params - avg[None, :]) / std[None, :]

train_params = torch.from_numpy(train_params.astype(np.float32)).to(device=device)
train_hists = torch.from_numpy(train_hists.astype(np.float32)).to(device=device)
validation_params = torch.from_numpy(validation_params.astype(np.float32)).to(device=device)
validation_hists = torch.from_numpy(validation_hists.astype(np.float32)).to(device=device)

class MLPLayer(nn.Sequential) :
    def __init__(self, Nin, Nout, activation=nn.LeakyReLU) :
        super().__init__(OrderedDict([('linear', nn.Linear(Nin, Nout, bias=True)),
                                      ('activation', activation()),
                                     ]))

class MLP(nn.Sequential) :
    def __init__(self, Nin, Nout, Nlayers=4, Nhidden=512) :
        # output is manifestly positive so we use ReLU in the final layer
        super().__init__(*[MLPLayer(Nin if ii==0 else Nhidden,
                                    Nout if ii==Nlayers else Nhidden,
                                    activation=(nn.LeakyReLU if ii!=Nlayers else nn.ReLU))
                           for ii in range(Nlayers+1)])

class Loss(nn.Module) :
    def __init__(self) :
        super().__init__()
    def forward(self, pred, targ) :
        # inputs have shape [batch, bin]

        # to avoid divergence (because pred could very well be exactly zero),
        # add a small term
        pred_ = pred + 1e-8

        # negative log-likelihood under poisson distribution assumption
        return - torch.sum(torch.special.xlogy(targ, pred_) - pred_ - torch.special.gammaln(1.0+targ)) \
               / pred.shape[0]

train_set = TensorDataset(train_params, train_hists)
train_loader = DataLoader(train_set, batch_size=256)

loss = Loss()
model = MLP(train_params.shape[1], train_hists.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

for ii in range(400) :
    model.train()
    for jj, (x, y) in enumerate(train_loader) :
        optimizer.zero_grad()
        ypred = model(x)
        l = loss(ypred, y)
        l.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    ypred = model(train_params)
    ltrain = loss(ypred, train_hists)

    ypred = model(validation_params)
    lvalidation = loss(ypred, validation_hists)

    print(f'iteration {ii:4}: {ltrain.item():8.2f}\t{lvalidation.item():8.2f}')

model.eval()
ypred = model(validation_params)
np.savez('vsf_mlp_test.npz',
         truth=validation_hists.detach().cpu().numpy(),
         prediction=ypred.detach().cpu().numpy())

torch.save(model.state_dict(), 'vsf_mlp.pt')
