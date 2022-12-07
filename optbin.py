from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available() :
    device = 'cuda'
else :
    device = 'cpu'

torch.manual_seed(42)

# where we care
RMIN = 30
RMAX = 80
ZMIN = 0.42
ZMAX = 0.68

with np.load('cdf_norm.npz') as f :
    norm_avg = f['avg']
    norm_std = f['std']
norm_avg = torch.from_numpy(norm_avg.astype(np.float32)).to(device=device)
norm_std = torch.from_numpy(norm_std.astype(np.float32)).to(device=device)

variations = [(0.0223, 0.0002),
              (0.1210, 0.0010),
              (1.0410, 0.0004),
              (3.0500, 0.0200),
              (0.9600, 0.0040),
              (0.2000, 0.1000), # Mnu
              (1.3000, 1.0000),
              (-0.400, 0.5000),
              (12.720, 0.0500),
              (0.3000, 0.1000),
              (14.000, 1.0000),
              (14.100, 1.0000),
              (0.7000, 0.3000),
              (7.5000, 1.5000),
              (0.0000, 1.0000),
              (-8.000, 2.0000),
              (5.0000, 20.000),]

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

class CDFModel(nn.Module) :
    def __init__(self, Nin, Nlayers=4, Nhidden=256) :
        super().__init__()
        self.mlp1 = MLP(Nin, Nhidden, Nlayers, Nhidden, out_positive=False)
        self.mlp2 = MLP(self.mlp1.Nout+2, 1, Nlayers, Nhidden, out_positive=True)
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
        edge1 = torch.clone(edge).reshape(-1, 2)
        edge1[:, 0] = CDFModel._norm_edge(edge1[:, 0], ZMIN, ZMAX)
        edge1[:, 1] = CDFModel._norm_edge(edge1[:, 1], RMIN, RMAX)
        x = torch.cat([torch.unsqueeze(x, 1).expand(-1, edge1.shape[0], -1),
                       torch.unsqueeze(edge1, 0).expand(x.shape[0], -1, -1)],
                       dim=-1)
        x = self.mlp2(x).reshape(batch, *ld)
        return x + 1e-8 # ensure positive

class HistModel(nn.Module) :
    # to compare to previous performance
    def __init__(self, *args, **kwargs) :
        super().__init__()

        self.cdfmodel = CDFModel(*args, **kwargs)

        # create checkerboard pattern for histogram evaluation
        self.weight = torch.tensor([[1, -1], [-1, 1]], requires_grad=False, dtype=torch.float32, device=device)\
                        .unsqueeze(0).unsqueeze(0)

    def __call__(self, x, zedges, Redges) :
        
        edges = torch.stack(torch.meshgrid(zedges, Redges,indexing='ij'), dim=-1)
        x = self.cdfmodel(x, edges) # shape [batch, z, R]

        # now apply the convolution to get bin counts, constrain to positive non-zero
        x = F.relu(F.conv2d(x.unsqueeze(1), self.weight).squeeze(1)) + 1e-8

        # shape [batch, z, R]
        return x

class Binning(nn.Module) :
    
    def __init__(self, lo, hi, N) :
        # produces N bins
        self.lo = lo
        self.hi = hi
        self.N = N

        super().__init__()
        self.register_parameter('x', nn.Parameter(torch.linspace(-4.0, 4.0, N+1, requires_grad=True)[1:-1]))
#        self.register_parameter('x', nn.Parameter(torch.linspace(-2.0, 2.0, N+1, requires_grad=True)[1:-1]))

    def forward(self) :
        x = self.lo + (self.hi-self.lo) * torch.sigmoid(torch.sort(self.x).values)
#        x = self.lo + (self.hi-self.lo) * 0.5 * (1+torch.erf(torch.sort(self.x).values))
        return torch.cat([torch.tensor([self.lo, ], device=device), x, torch.tensor([self.hi, ], device=device)])


class Fisher(nn.Module) :

    def __init__(self, Nz, NR) :
        
        super().__init__()
        self.emulator = HistModel(17)
        self.emulator.load_state_dict(torch.load('cdf_model.pt', map_location=device))
        for param in self.emulator.parameters() :
            param.requires_grad = False

        self.z_binning = Binning(ZMIN, ZMAX, Nz)
        self.R_binning = Binning(50.0, RMAX, NR)

        self.theta_fid = torch.tensor([v[0] for v in variations], requires_grad=False, device=device)
        self.theta_max_step = torch.tensor([v[1] for v in variations], requires_grad=False, device=device)

        self.indicators = torch.eye(len(self.theta_fid), device=device) # 17 x 17

        self.theta_fid_norm = (self.theta_fid - norm_avg) / norm_std

        cosmo_cov = np.loadtxt('/tigress/lthiele/mu_cov_plikHM_TTTEEE_lowl_lowE.dat', skiprows=3)
        cosmo_F = torch.from_numpy(np.linalg.inv(cosmo_cov).astype(np.float32)).to(device=device)
        self.prior = torch.zeros((len(self.theta_fid), len(self.theta_fid)), requires_grad=False, device=device)
        self.prior[:len(cosmo_cov), :len(cosmo_cov)] = cosmo_F

        # to stabilize numerics
        fake_sigma = 10.0 * self.theta_max_step[len(cosmo_cov):]
        self.prior[len(cosmo_cov):, len(cosmo_cov):] = torch.diagflat(1.0/fake_sigma**2)

    def forward(self, x) :
        # x should be of shape [batch, 17] and in 0, 1, contains the variations to use

        # [batch, 17]
        delta_theta = x * self.theta_max_step[None, :]

        # these are [batch, parameter, parameter]
        theta_p = self.theta_fid[None, None, :] + self.indicators[None, :, :] * delta_theta[:, None, :]
        theta_m = self.theta_fid[None, None, :] - self.indicators[None, :, :] * delta_theta[:, None, :]

        theta_p_norm = (theta_p - norm_avg[None, None, :]) / norm_std[None, None, :]
        theta_m_norm = (theta_m - norm_avg[None, None, :]) / norm_std[None, None, :]

        ld = theta_p_norm.shape[:2]

        zedges = self.z_binning()
        Redges = self.R_binning()

        d1 = len(zedges) - 1
        d2 = len(Redges) - 1

        # mu_hi, mu_lo are shape [batch, parameter, d1*d2]
        mu_hi = self.emulator(theta_p_norm.reshape(-1, 17), zedges, Redges).reshape(*ld, -1)
        mu_lo = self.emulator(theta_m_norm.reshape(-1, 17), zedges, Redges).reshape(*ld, -1)
        mu_fid = self.emulator(self.theta_fid_norm.reshape(-1, 17), zedges, Redges).reshape(-1)
        assert len(mu_fid) == d1 * d2

        # shape [batch, parameter, d1*d2]
        dmudtheta = 0.5 * (mu_hi - mu_lo) / delta_theta[:, :, None]

        # compute the Fisher matrices, shape [batch, theta, theta]
        Fab = torch.sum((dmudtheta[:, :, None, :] * dmudtheta[:, None, :, :]) / mu_fid[None, None, None, :],
                        dim=-1)

        # add the prior
        Fab = Fab + self.prior[None, :, :]

        return Fab

class Loss(nn.Module) :
    def __init__(self) :
        super().__init__()
    def forward(self, Fab) :
        # Fab is shape [batch, theta, theta]

#        return - torch.mean(torch.linalg.det(Fab))

        # compute the covariance matrices, shape [batch, theta, theta]
        Cab = torch.linalg.inv(Fab)

        # extract the error on Mnu, shape batch
        var_Mnu = torch.diagonal(Cab, dim1=1, dim2=2)[:, 5]
        print(f'sigma_Mnu={torch.mean(torch.sqrt(var_Mnu)).item()}')

        # for diagnostics
        var_logMmin = torch.diagonal(Cab, dim1=1, dim2=2)[:, 8]
        print(f'sigma_logMmin={torch.mean(torch.sqrt(var_logMmin)).item()}')
        var_mu_Mmin = torch.diagonal(Cab, dim1=1, dim2=2)[:, 15]
        print(f'sigma_mu_Mmin={torch.mean(torch.sqrt(var_mu_Mmin)).item()}')

        return torch.mean(torch.sqrt(var_Mnu))

loss = Loss()
model = Fisher(3, 32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3) 


batch_size = 256

orig_z = model.z_binning()
orig_R = model.R_binning()

for epoch in range(1000) :
    
    model.train()
    optimizer.zero_grad()
    x = 0.1 * (0.5 + 0.5 * torch.rand((batch_size, 17), device=device))
    Fab = model(x)
    l = loss(Fab)
    l.backward()
    optimizer.step()
    scheduler.step()

print(orig_z)
print(model.z_binning())
print(orig_R)
print(model.R_binning())
