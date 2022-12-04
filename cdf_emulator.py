from sys import argv
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataSet, DataLoader, TensorDataset

if torch.cuda.is_available() :
    device = 'cuda'
else :
    device = 'cpu'

torch.manual_seed(42)
rng = np.random.default_rng(42)

VALIDATION_FRAC = 0.2

# where we care
RMIN = 30
RMAX = 80
ZMIN = 0.42
ZMAX = 0.68

data_file = argv[1]

with np.load(data_file) as f :
    param_names = list(f['param_names'])
    cosmo_indices = f['cosmo_indices']
    params = f['params']
    radii = f['radii']
    redshifts = f['redshifts']

N = len(params)
validation_select = rng.choice([True, False], size=N, p=[VALIDATION_FRAC, 1.0-VALIDATION_FRAC])

train_x = params[~validation_select]
validation_x = params[validation_select]

train_z = redshifts[~validation_select]
validation_z = redshifts[validation_select]
train_R = radii[~validation_select]
validation_R = radii[validation_select]

test_zedges = np.array([ZMIN, 0.53, ZMAX])
test_Redges = np.linspace(RMIN, RMAX, num=33)
train_hists = np.stack([np.histogram2d(z_, R_, bins=[test_zedges, test_Redges])[0]
                        for z_, R_ in zip(train_z, train_R)])
validation_hists = np.stack([np.histogram2d(z_, R_, bins=[test_zedges, test_Redges])[0]
                            for z_, R_ in zip(validation_z, validation_R)])
train_hists = torch.from_numpy(train_hists.astype(np.float32)).to(device=device)
validation_hists = torch.from_numpy(validation_hists.astype(np.float32)).to(device=device)

# mask giving the real voids
train_not_fake = (train_z>0) * (train_R>0)

avg = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x = (train_x - avg[None, :]) / std[None, :]
validation_x = (validation_x - avg[None, :]) / std[None, :]

np.savez('cdf_norm.npz', avg=avg, std=std)

train_x = torch.from_numpy(train_x.astype(np.float32)).to(device=device)
validation_x = torch.from_numpy(validation_x.astype(np.float32)).to(device=device)
train_z = torch.from_numpy(train_z.astype(np.float32)).to(device=device)
validation_z = torch.from_numpy(validation_z.astype(np.float32)).to(device=device)
train_R = torch.from_numpy(train_R.astype(np.float32)).to(device=device)
validation_R = torch.from_numpy(validation_R.astype(np.float32)).to(device=device)

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
                                    activation=(nn.LeakyReLU if ii!=Nlayers and out_positive else nn.ReLU))
                           for ii in range(Nlayers+1)])

class CDFModel(nn.Module) :
    def __init__(self, Nin, Nlayers=4, Nhidden=512) :
        super().__init__()
        self.mlp1 = MLP(Nin, Nhidden, Nlayers, Nhidden, out_positive=False)
        self.mlp2 = MLP(self.mlp1.Nout+2, 1, Nlayers, Nhidden)
    @staticmethod
    def _norm_edge(x, lo, hi) :
        return (x-0.5*(lo+hi))/(hi-lo)
    def forward(self, x, edge) :
        # x ... [batch, 17]
        # edge ... [... , 2] (z, R) in this order
        # output is [batch, ...] (no singleton)
        batch = x.shape[0]
        x = self.mlp1(x) # output [batch, latent]
        ld = edge.shape[:-1]
        edge = edge.reshape(-1, 2)
        edge[:, 0] = CDFModel._norm_edge(edge[:, 0], ZMIN, ZMAX)
        edge[:, 1] = CDFModel._norm_edge(edge[:, 1], RMIN, RMAX)
        x = torch.cat([torch.unsqueeze(x, 1).expand(-1, edge.shape[0], -1),
                       torch.unsqueeze(edge, 0).expand(x.shape[0], -1, -1)],
                       dim=-1)
        x = self.mlp2(x).reshape(batch, *ld)
        return x + 1e-8 # avoid zero below

class Loss(nn.Module) :
    # implements a Poisson likelihood, applicable to any count data
    def __init__(self, norm_until) :
        super().__init__()
        self.norm_until = norm_until
    def forward(self, pred, targ) :
        # input shapes are arbitrary but identical
        x = - torch.sum(torch.special.xlogy(targ, pred) - pred - torch.special.gammaln(1.0+targ))
        if self.norm_until is not None :
            x /= np.prod(pred.shape[:self.norm_until])
        return x

class HistModel :
    # to compare to previous performance
    def __init__(self, zedges, Redges) :
        zedges = torch.from_numpy(zedges.astype(np.float32)).to(device=device)
        Redges = torch.from_numpy(Redges.astype(np.float32)).to(device=device)

        # shape [z, R, 2]
        self.edges = torch.stack(torch.meshgrid(zedges, Redges,indexing='ij'), dim=-1)

        # create checkerboard pattern for histogram evaluation
        self.weight = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32, device=device)\
                        .unsqueeze(0).unsqueeze(0)

    def __call__(self, cdfmodel, x) :
        x = cdfmodel(x, self.edges) # shape [batch, z, R]

        # now apply the convolution to get bin counts, constrain to positive non-zero
        x = F.relu(F.conv2d(x.unsqueeze(1), self.weight).squeeze(1)) + 1e-8

        # shape [batch, z, R]
        return x

def epoch_data(idx, Npoints) :
    rng_ = np.random.default_rng(idx)
    zedges = torch.from_numpy(rng_.uniform(ZMIN, ZMAX, size=Npoints).astype(np.float32)).to(device=device)
    Redges = torch.from_numpy(rng_.uniform(RMIN, RMAX, size=Npoints).astype(np.float32)).to(device=device)
    valid_R = train_R[:, None, :] < Redges[None, :, None]
    valid_z = train_z[:, None, :] < zedges[None, :, None]
    all_valid = valid_R * valid_z * train_not_fake[:, None, :] # shape [batch, Npoints, void index]
    target = torch.sum(all_valid, dim=-1).to(dtype=torch.float32) # shape [batch, Npoints]
    edges = torch.stack([zedges, Redges], -1)
    return edges, target

train_loss = Loss()
test_loss = Loss(norm_until=2)
model = CDFModel(params.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

hist_model = HistModel(test_zedges, test_Redges)

for epoch in range(100) :
    model.train()
    edge, target = epoch_data(epoch, 64)
    train_set = TensorDataset(train_x, target)
    train_loader = DataLoader(train_set, batch_size=256)
    for jj, (x, y) in enumerate(train_loader) :
        optimizer.zero_grad()
        pred = model(x, edge)
        l = train_loss(pred, y)
        l.backward()
        optimizer.step()

    model.eval()
    pred = hist_model(model, train_x)
    ltrain = test_loss(pred, train_hists)

    pred = hist_model(model, validation_x)
    lvalidation = test_loss(pred, validation_hists)

    print(f'iteration {epoch:4}: {ltrain.item():8.2f}\t{lvalidation.item():8.2f}')
