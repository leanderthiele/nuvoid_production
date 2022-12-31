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

torch.manual_seed(43)
rng = np.random.default_rng(42)

VALIDATION_FRAC = 0.2

KMIN = 0.025
KMAX = 0.12

with np.load('/tigress/lthiele/plk_grouped_data.npz') as fp :
    cosmo_indices = fp['cosmo_indices']
    params = fp['params']
    p0k_samples = fp['p0k']
with np.load('/tigress/lthiele/boss_dr12/plk/boss_plk.npz') as fp :
    k = fp['k']
    boss_p0k = fp['p0k']
approx_cov = np.load('/tigress/lthiele/boss_dr12/plk/approx_cov.npy')
print(f'Have {len(params)} samples')

# low k are weird, so we exclude them to avoid instability
min_idx = np.argmin(np.fabs(k-KMIN))
max_idx = np.argmin(np.fabs(k-KMAX))+1
p0k_samples = p0k_samples[..., min_idx:max_idx]
k = k[min_idx:max_idx]
boss_p0k = boss_p0k[min_idx:max_idx]
approx_cov = approx_cov[min_idx:max_idx][:, min_idx:max_idx]

# for better normalization
p0k_samples = p0k_samples * k[None, None, :]
boss_p0k = boss_p0k * k
approx_cov = approx_cov * (k[:, None] * k[None, :])

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

# these are [parameter, sample, k]
train_p0k = p0k_samples[~validation_select]
validation_p0k = p0k_samples[validation_select]

# first try: only train on the averages, this should still be enough samples hopefully
#p0k_target = np.mean(p0k_samples, axis=1)
#approx_cov /= p0k_samples.shape[1]

# keep the individual samples in the training set to increase size
train_p0k_var = np.var(train_p0k, axis=1).repeat(p0k_samples.shape[1], axis=0)
train_p0k = train_p0k.reshape(-1, len(k))
train_x = train_x.repeat(p0k_samples.shape[1], axis=0)

# for validation, we look at the means
validation_p0k_var = np.var(validation_p0k, axis=1)
validation_p0k = np.mean(validation_p0k, axis=1)

# precision matrix
covinv = np.linalg.inv(approx_cov)

# now to torch
train_x = torch.from_numpy(train_x.astype(np.float32)).to(device=device)
validation_x = torch.from_numpy(validation_x.astype(np.float32)).to(device=device)
train_p0k = torch.from_numpy(train_p0k.astype(np.float32)).to(device=device)
train_p0k_var = torch.from_numpy(train_p0k_var.astype(np.float32)).to(device=device)
validation_p0k = torch.from_numpy(validation_p0k.astype(np.float32)).to(device=device)
validation_p0k_var = torch.from_numpy(validation_p0k_var.astype(np.float32)).to(device=device)
covinv = torch.from_numpy(covinv.astype(np.float32)).to(device=device)
boss_p0k = torch.from_numpy(boss_p0k.astype(np.float32)).to(device=device)

class MLPLayer(nn.Sequential) :
    def __init__(self, Nin, Nout, activation=nn.LeakyReLU) :
        super().__init__(OrderedDict([('linear', nn.Linear(Nin, Nout, bias=True)),
                                      ('activation', activation()),
                                     ]))

class MLP(nn.Sequential) :
    def __init__(self, Nin, Nout, Nlayers=16, Nhidden=128, out_positive=True) :
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
        self.mse_loss = nn.MSELoss(reduction='mean')
    def forward(self, pred, targ, var) :

        #targ_ratio = boss_p0k[None, :] / targ
        #scaled_covinv = covinv[None, :, :] * (targ_ratio[:, :, None] * targ_ratio[:, None, :])

        # we do not care that much about places where the fit is super bad anyways,
        # so we reduce their contribution to the loss function
        delta_boss = targ - boss_p0k[None, :]
        chisq_boss = torch.einsum('bi,ij,bj->b', delta_boss, covinv, delta_boss)
        #chisq_boss = torch.sum(delta_boss**2 / var, dim=1)
        score = torch.exp(-chisq_boss/1e4)

        # shapes are [batch, Nk]
        # return self.mse_loss(torch.log(pred+1e-8), torch.log(targ+1e-8))
        # return self.mse_loss(pred, targ)

        delta = pred-targ
        #chisq = torch.einsum('bi,bij,bj->b', delta, scaled_covinv, delta)
        chisq = torch.sum(delta**2 / var, dim=1)
        return torch.mean(score * chisq), score

EPOCHS = 50

validation_set = TensorDataset(validation_x, validation_p0k, validation_p0k_var)
validation_loader = DataLoader(validation_set, batch_size=512)
loss = Loss()
model = MLP(train_x.shape[-1], train_p0k.shape[-1], out_positive=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=EPOCHS, verbose=True)

for epoch in range(EPOCHS) :
    
    # add some noise to prevent overfitting
    train_set = TensorDataset(train_x+torch.normal(0.0, 0.1, size=train_x.shape, device=device),
                              train_p0k, train_p0k_var)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)

    model.train()
    ltrain = []
    for x, y, var in train_loader :
        optimizer.zero_grad()
        pred = model(x)
        l, _ = loss(pred, y, var)
        ltrain.append(l.item())
        l.backward()
        optimizer.step()
    scheduler.step()
    ltrain = np.mean(np.array(ltrain))

    model.eval()
    lvalidation = []
    for x, y, var in validation_loader :
        pred = model(x)
        l, _ = loss(pred, y, var)
        lvalidation.append(l.item())
    lvalidation = np.mean(np.array(lvalidation))

    print(f'iteration {epoch:4}: {ltrain:16.2f}\t{lvalidation:16.2f}')

model.eval()
predictions = []
truth = []
scores = []
for x, y, var in validation_loader :
    pred = model(x)
    predictions.extend(pred.detach().cpu().numpy())
    truth.extend(y.cpu().numpy())
    _, score = loss(pred, y, var)
    scores.extend(score.cpu().numpy())
predictions = np.array(predictions)
truth = np.array(truth)
scores = np.array(scores)
sorter = np.argsort(scores)[::-1]
predictions = predictions[sorter]
truth = truth[sorter]
scores = scores[sorter]
np.savez('plk_validation_test.npz', predictions=predictions, truth=truth, scores=scores)
