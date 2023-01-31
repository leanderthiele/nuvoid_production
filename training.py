from sys import argv
import numpy as np

import torch
import torch.nn as nn

from mlp import MLP
from traindata import TrainData

BATCH_SIZE = 256
EPOCHS = 100
MAX_LR = 1e-2
CHISQ_CUT = 1e3 # 90% of chisq is <1e3, 96% <1e4, 98% <1e5

class Loss(nn.Module) :
    """ in the first approximation, we expect the covariance to be the identity """

    def __init__ (self, chisq_cut) :
        self.chisq_cut = chisq_cut
        super().__init__()

    def forward (self, pred, targ, chisq) :
        delta = pred - targ
        return torch.mean(torch.exp(-chisq/self.chisq_cut)[:, None] * torch.square(delta))

version = int(argv[1])
compression_hash = argv[2]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

model = MLP(17, 17, Nlayers=4, Nhidden=512).to(device)
traindata = TrainData(version, compression_hash, device, batch_size=BATCH_SIZE)
loss = Loss(CHISQ_CUT)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=EPOCHS, verbose=True)

for epoch in range(EPOCHS) :
    
    model.train()
    ltrain = []
    for x, y, c in traindata.train_loader :
        optimizer.zero_grad()
        pred = model(x)
        l = loss(pred, y, c)
        ltrain.append(l.item())
        l.backward()
        optimizer.step()
    scheduler.step()
    print(ltrain)
    ltrain = np.mean(np.array(ltrain))

    model.eval()
    lvalidation = []
    for x, y, c in traindata.validation_loader :
        pred = model(x)
        l = loss(pred, y, c)
        lvalidation.append(l.item())
    lvalidation = np.mean(np.array(lvalidation))

    print(f'iteration {epoch:4}: {ltrain:16.2f}\t{lvalidation:16.2f}')
