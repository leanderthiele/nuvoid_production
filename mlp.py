from collections import OrderedDict

import torch
import torch.nn as nn

class MLPLayer(nn.Sequential) :
    
    def __init__ (self, Nin, Nout, activation=nn.LeakyReLU, dropout=None) :
        super().__init__(OrderedDict([('linear', nn.Linear(Nin, Nout, bias=True)),
                                      ('activation', activation()),
                                      ('dropout', nn.Dropout(p=dropout) if dropout else nn.Identity()),
                                     ]))

class MLP(nn.Sequential) :
    
    def __init__ (self, Nin, Nout, Nlayers=4, Nhidden=512, dropout=0.1) :
        
        self.Nin = Nin
        self.Nout = Nout
        super().__init__(*[MLPLayer(Nin if ii==0 else Nhidden,
                                    Nout if ii==Nlayers else Nhidden,
                                    activation=nn.LeakyReLU if ii!=Nlayers else nn.Identity,
                                    dropout=None if ii==0 or ii==Nlayers else dropout)
                           for ii in range(Nlayers+1)])
