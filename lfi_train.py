from sys import argv
import pickle
from datetime import datetime
import hashlib

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import sbi
import sbi.inference
from sbi import utils as sbi_utils

from read_txt import read_txt

SETTINGS = dict(
                method='SNRE',
                model=('resnet', {'hidden_features': 128, 'num_blocks': 2}),
                consider_params=['Mnu', 'hod_log_Mmin', 'hod_mu_Mmin', ],
               )

ident = hashlib.md5(f'{SETTINGS}'.encode('utf-8')).hexdigest()

filebase = '/tigress/lthiele/nuvoid_production'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

version = int(argv[1])
compression_hash = argv[2]

data_fname = f'{filebase}/datavectors_trials.npz'
compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'
model_fname = f'{filebase}/lfi_model_v{version}_{compression_hash}_{ident}.sbi'
tb_logdir = f'{filebase}/sbi_logs_v{version}_{compression_hash}/{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_{ident}'

if 'NRE' in SETTINGS['method'] :
    arch = 'classifier_nn'
elif 'NPE' in SETTINGS['method'] :
    arch = 'posterior_nn'
elif 'NLE'in SETTINGS['method'] :
    arch = 'likelihood_nn'
# this returns a method
make_model = getattr(sbi_utils.get_nn_models, arch)(model=SETTINGS['model'][0],
                                                    z_score_theta='independent',
                                                    z_score_x='independent',
                                                    **SETTINGS['model'][1])
inference = getattr(sbi.inference, SETTINGS['method'])(classifier=make_model,
                                                       device=device,
                                                       show_progress_bars=True,
                                                       summary_writer=SummaryWriter(log_dir=tb_logdir))


normalization = read_txt(compress_fname, 'normalization:')
compression_matrix = read_txt(compress_fname, 'compression matrix:')
with np.load(data_fname) as f :
    data = f['data'].astype(np.float64)
    param_names = list(f['param_names'])
    params = f['params']
data = np.einsum('ab,ib->ia', compression_matrix, data/normalization[None, :])
param_indices = [params.index(s) for s in SETTINGS['consider_params']]

theta = torch.from_numpy(params[:, param_indices].astype(np.float32)).to(device=device)
x = torch.from_numpy(data.astype(np.float32)).to(device=device)

inference = inference.append_simulations(theta=theta, x=x)
density_estimator = inference.train(max_num_epochs=200)
posterior = inference.build_posterior(density_estimator)

with open(model_fname, 'w') as f :
    f.write(f'{SETTINGS}\n')
with open(model_fname, 'ab') as f :
    pickle.dump(posterior, f)
