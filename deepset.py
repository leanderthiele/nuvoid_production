from sys import argv

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from capped_mean import capped_mean

datafile = argv[1]

VALIDATION_FRAC = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with np.load(datafile) as f :
    param_names = f['param_names']
    cosmo_indices = f['cosmo_indices']
    params = f['params']
    radii = f['radii']
    redshifts = f['redshifts']

# number of voids per sample
Nvoids = np.count_nonzero(radii>0, axis=1)

# sanity checks
assert np.all(Nvoids == np.count_nonzero(redshifts>0, axis=1))
assert all(np.all(x[:n]>0) for x, n in zip(radii, Nvoids))
assert all(np.all(x[:n]>0) for x, n in zip(redshifts, Nvoids))

# shape [Nsamples, void index, 2]
data = np.stack([radii, redshifts], axis=-1)

rng = np.random.default_rng(42)
validation_select = rng.choice([True, False], size=len(radii), p=[VALIDATION_FRAC, 1-VALIDATION_FRAC])

TARGET_IDX = 5 # M_nu

train_params = params[~validation_select]
train_data = data[~validation_select]
train_Nvoids = Nvoids[~validation_select]
validation_params = params[validation_select]
validation_data = data[validation_select]
validation_Nvoids = Nvoids[validation_select]

# to torch
train_params = torch.from_numpy(train_params.astype(np.float32)).to(device=device)
train_data = torch.from_numpy(train_data.astype(np.float32)).to(device=device)
train_Nvoids = torch.from_numpy(train_Nvoids.astype(np.int32)).to(device=device)
validation_params = torch.from_numpy(validation_params.astype(np.float32)).to(device=device)
validation_data = torch.from_numpy(validation_data.astype(np.float32)).to(device=device)
validation_Nvoids = torch.from_numpy(validation_Nvoids.astype(np.int32)).to(device=device)

train_target = train_params[:, TARGET_IDX].unsqueeze(-1)
validation_target = validation_params[:, TARGET_IDX].unsqueeze(-1)

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

class DeepSetLayer(nn.Module) :
    def __init__(self, Nin, Nout, other_dim, *args, **kwargs) :
        # other_dim is for potentially other input that gets concatenated as context
        # args, kwargs get passed to MLP constructor
        self.other_dim = other_dim
        super().__init__()
        self.mlp = MLP(Nin + (0 if other_dim is None else other_dim), Nout, *args, **kwargs)
    def forward(self, x, N, other=None) :
        # x has shape [batch, particle, features]
        # N has shape [batch]
        # other, if given, has shape [batch, features]
        assert x.shape[0] == N.shape[0]
        if other is not None :
            assert other.shape[-1] == self.other_dim
            assert x.shape[0] == other.shape[0]
            x = torch.cat([x, torch.unsqueeze(other, 1).expand(-1, x.shape[1], -1)], dim=-1)
        # transform to shape [batch, particle, feature]
        x = self.mlp(x)
        # take the reduction operation to [batch, feature]
        x = capped_mean(x, N)
        return x

class DeepSet(nn.Module) :
    def __init__(self, Nin, Nout, Ncontext, *args, **kwargs) :
        # Ncontext should be a list of intermediate context dimensions, or None if only a single DeepSetLayer
        # should be used
        self.Nin = Nin
        self.Nout = Nout
        self.Ncontext = list() if Ncontext is None else Ncontext
        self.Nlayers = len(Ncontext)+1
        super().__init__()
        self.layers = nn.ModuleList([DeepSetLayer(Nin,
                                                  Nout if ii==self.Nlayers-1 else Ncontext[ii],
                                                  other_dim=None if ii==0 else Ncontext[ii-1],
                                                  *args, **kwargs)
                                     for ii in range(self.Nlayers)])
    def forward(self, x, N) :
        latent = None
        for l in self.layers :
            latent = l(x, N, latent)
        return latent

class Loss(nn.Module) :
    # simple MSE at the moment
    def __init__(self) :
        super().__init__()
        self.l = nn.MSELoss(reduction='mean')
    def forward(self, pred, targ) :
        return self.l(pred, targ)

EPOCHS = 100

loss = Loss()
model = DeepSet(data.shape[-1], 1, None).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_set = TensorDataset(train_data, train_Nvoids, train_target)
train_loader = DataLoader(train_set, batch_size=256)

for epoch in range(EPOCHS) :
    
    model.train()
    for ii, (x, n, y) in enumerate(train_loader) :
        optimizer.zero_grad()
        pred = model(x, n)
        l = loss(pred, y)
        l.backward()
        optimizer.step()

    model.eval()

    pred = model(train_data, train_Nvoids)
    ltrain = loss(pred, train_target)

    pred = model(validation_data, validation_Nvoids)
    lvalidation = loss(pred, validation_target)

    print(f'iteration {epoch:4}: {ltrain.item():8.2f}\t{lvalidation.item():8.2f}')
