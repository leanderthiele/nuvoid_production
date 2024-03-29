from sys import argv
import numpy as np

import torch
import torch.nn as nn

from mlp import MLP
from traindata import TrainData

filebase = '/tigress/lthiele/nuvoid_production'

USE_AVG = True # whether we use the realization-averaged data
BATCH_SIZE = 16
EPOCHS = 100
MAX_LR = 1e-3
CHISQ_CUT = 1e3 # 90% of chisq is <1e3, 96% <1e4, 98% <1e5
NOISE = None
WEIGHT_DECAY = 1e-2
DROPOUT = None

class Loss(nn.Module) :
    """ in the first approximation, we expect the covariance to be the identity """

    def __init__ (self, chisq_cut) :
        self.chisq_cut = chisq_cut
        super().__init__()

    def forward (self, pred, targ, chisq, nsims) :
        delta = pred - targ
        return torch.mean((nsims * torch.exp(-chisq/self.chisq_cut))[:, None] * torch.square(delta))

version = int(argv[1])
compression_hash = argv[2]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

model = MLP(17, 17, Nlayers=4, Nhidden=4096, dropout=DROPOUT).to(device)
traindata = TrainData(version, compression_hash, device, batch_size=BATCH_SIZE, use_avg=USE_AVG)
loss = Loss(CHISQ_CUT)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=EPOCHS, verbose=True)

for epoch in range(EPOCHS) :
    
    model.train()
    ltrain = []
    for x, y, c, n in traindata.train_loader :
        if NOISE is not None :
            x += torch.normal(0.0, NOISE, x.shape, device=device)
        optimizer.zero_grad()
        pred = model(x)
        l = loss(pred, y, c, n)
        ltrain.append(l.item())
        l.backward()
        optimizer.step()
    scheduler.step()
    ltrain = np.mean(np.array(ltrain))

    model.eval()
    lvalidation = []
    for x, y, c, n in traindata.validation_loader :
        pred = model(x)
        l = loss(pred, y, c, n)
        lvalidation.append(l.item())
    lvalidation = np.mean(np.array(lvalidation))

    print(f'iteration {epoch:4}: {ltrain:16.2f}\t{lvalidation:16.2f}')

# save the model and relevant information to file
to_save = dict(model_state=model.state_dict(),
               # for convenience, this dict can be passed as kwargs to MLP constructor
               model_meta={'Nin': model.Nin, 'Nout': model.Nout,
                           'Nlayers': model.Nlayers, 'Nhidden': model.Nhidden,
                           'dropout': model.dropout},
               input_params=TrainData.use_params,
               params_avg=torch.from_numpy(traindata.norm_avg.astype(np.float32)),
               params_std=torch.from_numpy(traindata.norm_std.astype(np.float32)),
              )
torch.save(to_save, f'{filebase}/mlp_v{version}_{compression_hash}.pt')
               

model.eval()
predictions = []
truth = []
chisq = []
for x, y, c, n in traindata.validation_loader :
    pred = model(x)
    predictions.extend(pred.detach().cpu().numpy())
    truth.extend(y.cpu().numpy())
    chisq.extend(c.cpu().numpy())
np.savez(f'validation_v{version}_{compression_hash}.npz',
         predictions=np.array(predictions),
         truth=np.array(truth),
         chisq=np.array(chisq))
