from collections import OrderedDict

import torch
import torch.nn as nn

class MLPLayer(nn.Sequential) :
    
    def __init__ (self, Nin, Nout, activation=nn.LeakyReLU) :
        super().__init__(OrderedDict([('linear': nn.Linear(Nin, Nout, bias=True)),
                                      ('activation': activation()),
                                     ]))

class MLP(nn.Sequential) :
    
    def __init__ (self, Nin, Nout, Nlayers=4, Nhidden=512) :
        
        self.Nin = Nin
        self.Nout = Nout
        super().__init__(*[MLPLayer(Nin if ii==0 else Nhidden,
                                    Nout if ii==Nlayers else Nhidden,
                                    activation=nn.LeakyReLU if ii!=Nlayers else nn.Identity)
                           for ii in range(Nlayers+1)])
