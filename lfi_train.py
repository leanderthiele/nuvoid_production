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
                model=('resnet', {
                                  'hidden_features': 128,
                                  'num_blocks': 2,
                                  #'dropout_probability': 0.6
                                 }
                      ),
                # TODO try changing this (keeping Mnu)
                consider_params=['Mnu', 'hod_log_Mmin', 'hod_mu_Mmin', ],
                priors = {
                          'Mnu': [0.0, 0.6],
                          'hod_transf_P1': [-3.0, 3.0],
                          'hod_abias': [-1.0, 1.0],
                          'hod_log_Mmin': [12.5, 13.2],
                          'hod_sigma_logM': [0.1, 0.8],
                          'hod_log_M0': [12.5, 15.5],
                          'hod_log_M1': [12.5, 15.5],
                          'hod_alpha': [0.2, 1.5],
                          'hod_transf_eta_cen': [5.0, 10.0],
                          'hod_transf_eta_sat': [-1.0, 1.0],
                          'hod_mu_Mmin': [-20.0, 20.0],
                          'hod_mu_M1': [-40.0, 40.0]
                         },
                # bs=16,
                lr=1e-2,
               )

ident = hashlib.md5(f'{SETTINGS}'.encode('utf-8')).hexdigest()
print(f'ident={ident}')

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
param_indices = [param_names.index(s) for s in SETTINGS['consider_params']]

theta = torch.from_numpy(params[:, param_indices].astype(np.float32)).to(device=device)
x = torch.from_numpy(data.astype(np.float32)).to(device=device)

inference = inference.append_simulations(theta=theta, x=x)
density_estimator = inference.train(max_num_epochs=200,
                                    training_batch_size=SETTINGS['bs'] if 'bs' in SETTINGS else 50,
                                    learning_rate=SETTINGS['lr'] if 'lr' in SETTINGS else 5e-4)
prior = sbi_utils.BoxUniform(low=torch.Tensor([SETTINGS['priors'][s][0] for s in SETTINGS['consider_params']]),
                             high=torch.Tensor([SETTINGS['priors'][s][1] for s in SETTINGS['consider_params']]),
                             device=device)
posterior = inference.build_posterior(density_estimator, prior=prior)

with open(model_fname, 'w') as f :
    f.write(f'{SETTINGS}\n')
with open(model_fname, 'ab') as f :
    pickle.dump(posterior, f)
