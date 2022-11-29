# Command line arguments:
#   [1] file containing the training data

from sys import argv
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

train_params = torch.from_numpy(train_params.astype(np.float32))
train_hists = torch.from_numpy(train_hists.astype(np.float32))
validation_params = torch.from_numpy(validation_params.astype(np.float32))
validation_hists = torch.from_numpy(validation_hists.astype(np.float32))

class MLPLayer(nn.Sequential) :
    def __init__(self, Nin, Nout, activation=nn.LeakyReLU) :
        super().__init__(OrderedDict([('linear', nn.Linear(Nin, Nout, bias=True)),
                                      ('activation', activation()),
                                     ]))

class MLP(nn.Sequential) :
    def __init__(self, Nin, Nout, Nlayers=4, Nhidden=32) :
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
        return - torch.sum(targ * torch.log(pred_) - pred_ - torch.special.gammaln(1.0+targ)) \
               / pred.shape[0]

train_set = TensorDataset(train_params, train_hists)
validation_set = TensorDataset(validation_params, validation_hists)

train_loader = DataLoader(train_set)
validation_loader = DataLoader(validation_set)

model = MLP(train_params.shape[1], train_hists.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = Loss()

for ii in range(10) :
    model.train()
    for jj, (x, y) in enumerate(training_loader) :
        optimizer.zero_grad()
        ypred = model(x)
        l = loss(ypred, y)
        l.backward()
        optimizer.step()

    model.eval()
    ypred = model(train_params)
    ltrain = loss(ypred, train_hists)

    ypred = model(validation_params)
    lvalidation = loss(ypred, validation_hists)

    print(f'{ltrain.item()}\t{lvalidation.item()}')
