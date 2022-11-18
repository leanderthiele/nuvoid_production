# Command line arguments:
#   [1] file containing the training data

from sys import argv
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

VALIDATION_FRAC = 0.2
NUM_ROUND = 100

data_file = argv[1]

with np.load(data_file) as f :
    param_names = list(f['param_names'])
    params = f['params']
    values = f['values'] # the log-likelihood

# FIXME
# values = np.exp(values-np.max(values))

N = len(values)
assert N == params.shape[0]
assert len(param_names) == params.shape[1]

rng = np.random.default_rng(42)
validation_select = rng.choice([True, False], size=N, p=[VALIDATION_FRAC, 1.0-VALIDATION_FRAC])

# TODO maybe this helps
min_clip = np.max(values) - 100
values[values < min_clip] = min_clip

train_params = params[~validation_select]
train_values = values[~validation_select]
validation_params = params[validation_select]
validation_values = values[validation_select]

# to increase size of training set and learn what we are actually interested in,
# we concentrate on the *difference* in log-likelihoods
if False :
    train_param_diffs = (train_params[:, None, :] - train_params[None, :, :])\
                                   .reshape(-1, train_params.shape[1])
    train_values = (train_values[:, None] - train_values[None, :]).flatten()
    validation_param_diffs = (validation_params[:, None, :] - validation_params[None, :, :])\
                                  .reshape(-1, validation_params.shape[1])
    validation_values = (validation_values[:, None] - validation_values[None, :]).flatten()

    train_params = np.concatenate((np.repeat(train_params, train_params.shape[0], axis=0),
                                   train_param_diffs), axis=-1)
    validation_params = np.concatenate((np.repeat(validation_params, validation_params.shape[0], axis=0),
                                       validation_param_diffs), axis=-1)

train_params = torch.from_numpy(train_params)
train_values = torch.from_numpy(train_values)
validation_params = torch.from_numpy(validation_params)
validation_values = torch.from_numpy(validation_values)

class MLPLayer(nn.Sequential) :
    def __init__(self, Nin, Nout, activation=True) :
        super().__init__(OrderedDict([('linear', nn.Linear(N_in, N_out, bias=True)),
                                      ('activation': nn.LeakyReLU() if activation else nn.Identity()),
                                     ]))

class MLP(nn.Sequential) :
    def __init__(self, N_in, N_out, Nlayers=4, Nhidden=64) :
        super().__init__(*[MLPLayer(N_in if ii==0 else Nhidden,
                                    N_out if ii==Nlayers else Nhidden,
                                    activation=(ii != Nlayers))
                           for ii in range(Nlayers+1)])

training_loader = DataLoader(list(zip(train_params, train_values)), batch_size=64)
model = MLP(train_params.shape[1], 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = torch.nn.MSELoss(reduction='mean')
for ii in range(10) :
    model.train()
    for x, y in training_loader :
        optimizer.zero_grad()
        ypred = model(x)
        l = loss(y, ypred)
        l.backward()
        optimizer.step()

    model.eval()
    ypred = model(validation_params)
    l = loss(validation_values, ypred)
    print(l)
