from sys import argv
from collections import OrderedDict

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

rng = np.random.default_rng(42)
validation_select = rng.choice([True, False], size=len(radii), p=[VALIDATION_FRAC, 1-VALIDATION_FRAC])

# normalizations -- don't cheat!
train_radii = radii[~validation_select]
train_redshifts = redshifts[~validation_select]
train_N = Nvoids[~validation_select]
R_avg = np.mean(train_radii[train_radii>0])
R_std = np.std(train_radii[train_radii>0])
z_avg = np.mean(train_redshifts[train_redshifts>0])
z_std = np.std(train_redshifts[train_redshifts>0])
N_avg = np.mean(train_N.astype(float))
N_std = np.std(train_N.astype(float))

# shape [Nsamples, void index, 2]
data = np.stack([(radii-R_avg)/R_std, (redshifts-z_avg)/z_std], axis=-1)
norm_Nvoids = (Nvoids.astype(float) - N_avg) / N_std

# TARGET_IDX = 5 # M_nu
TARGET_IDX = 15 # mu_Mmin

train_params = params[~validation_select]
train_data = data[~validation_select]
train_Nvoids = Nvoids[~validation_select]
train_norm_Nvoids = norm_Nvoids[~validation_select]
validation_params = params[validation_select]
validation_data = data[validation_select]
validation_Nvoids = Nvoids[validation_select]
validation_norm_Nvoids = norm_Nvoids[validation_select]

# to torch
train_params = torch.from_numpy(train_params.astype(np.float32)).to(device=device)
train_data = torch.from_numpy(train_data.astype(np.float32)).to(device=device)
train_Nvoids = torch.from_numpy(train_Nvoids.astype(np.int32)).to(device=device)
train_norm_Nvoids = torch.from_numpy(train_norm_Nvoids.astype(np.float32)).to(device=device)
validation_params = torch.from_numpy(validation_params.astype(np.float32)).to(device=device)
validation_data = torch.from_numpy(validation_data.astype(np.float32)).to(device=device)
validation_Nvoids = torch.from_numpy(validation_Nvoids.astype(np.int32)).to(device=device).unsqueeze(-1)
validation_norm_Nvoids = torch.from_numpy(validation_norm_Nvoids.astype(np.float32)).to(device=device).unsqueeze(-1)

train_target = train_params[:, TARGET_IDX].unsqueeze(-1)
validation_target = validation_params[:, TARGET_IDX].unsqueeze(-1)

class MLPLayer(nn.Sequential) :
    def __init__(self, Nin, Nout, activation=True) :
        super().__init__(OrderedDict([('linear', nn.Linear(Nin, Nout, bias=True)),
                                      ('activation', nn.LeakyReLU() if activation else nn.Identity()),
                                     ]))

class MLP(nn.Sequential) :
    def __init__(self, Nin, Nout, Nlayers=2, Nhidden=256, last_activation=True) :
        # output is manifestly positive so we use ReLU in the final layer
        self.Nin = Nin
        self.Nout = Nout
        super().__init__(*[MLPLayer(Nin if ii==0 else Nhidden,
                                    Nout if ii==Nlayers else Nhidden,
                                    activation=ii!=Nlayers or last_activation)
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
    def __init__(self, Nin, Nout, Nlatent, Ncontext, Nds, *args, **kwargs) :
        self.Nin = Nin
        self.Nout = Nout
        self.Nlatent = Nlatent
        self.Ncontext = Ncontext
        self.Nds = Nds
        super().__init__()
        self.layers = nn.ModuleList([DeepSetLayer(Nin, self.Nlatent,
                                                  other_dim=None if ii==0 else self.Nlatent,
                                                  *args, **kwargs)
                                     for ii in range(self.Nds)])
        self.collapse = MLP(self.layers[-1].mlp.Nout+Ncontext, Nout, last_activation=False, *args, **kwargs)
    def forward(self, x, N, context) :
        # context is the global context which is added to the final MLP input, shape [batch, feature]
        latent = None
        for l in self.layers :
            latent = l(x, N, latent)
        # latent has shape [batch, features]
        latent = torch.cat([latent, context], dim=-1)
        return self.collapse(latent)

class Loss(nn.Module) :
    # simple MSE at the moment
    def __init__(self) :
        super().__init__()
        self.l = nn.MSELoss(reduction='mean')
    def forward(self, pred, targ) :
        return self.l(pred, targ)

EPOCHS = 100

loss = Loss()
model = DeepSet(data.shape[-1], 1, [16,]).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_set = TensorDataset(train_data, train_Nvoids, train_norm_Nvoids, train_target)
train_loader = DataLoader(train_set, batch_size=256)
validation_set = TensorDataset(validation_data, validation_Nvoids, validation_norm_Nvoids, validation_target)
validation_loader = DataLoader(validation_set, batch_size=2048)

for epoch in range(EPOCHS) :
    
    model.train()
    ltrain = []
    for ii, (x, n, nnorm, y) in enumerate(train_loader) :
        optimizer.zero_grad()
        pred = model(x, n, nnorm)
        l = loss(pred, y)
        ltrain.append(l.item())
        l.backward()
        optimizer.step()

    ltrain = np.sqrt(np.mean(np.array(ltrain)))

    model.eval()

    lvalidation = []
    for ii, (x, n, nnorm, y) in enumerate(validation_loader) :
        pred = model(x, n, nnorm)
        l = loss(pred, y)
        lvalidation.append(l.item())

    lvalidation = np.sqrt(np.mean(np.array(lvalidation)))

    print(f'iteration {epoch:4}: {ltrain:8.2f}\t{lvalidation:8.2f}')
