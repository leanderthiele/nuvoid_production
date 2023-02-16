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
from lfi_load_posterior import load_posterior

SETTINGS = dict(
                method='SNRE',
                model=('resnet', {
                                  'hidden_features': 256,
                                  'num_blocks': 2,
                                  #'dropout_probability': 0.6
                                 }
                      ),
                consider_params=['Mnu', 'hod_log_Mmin', 'hod_mu_Mmin', ],
                # consider_params=['Mnu', ],
                # consider_params=['Mnu', 'hod_transfP1', 'hod_sigma_logM', 'hod_transf_eta_cen', ],
                # consider_params=['Mnu', 'hod_abias', 'hod_log_M1', 'hod_alpha', 'hod_transf_eta_sat', ],
                priors={
                        'Mnu': [0.0, 0.6],
                        'hod_transfP1': [-3.0, 3.0],
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
                # bs=256,
                lr=3e-3,
                chisq_max=1e4,
                noise=1e-2, # eV
                one_cycle=True,
                optimizer_kwargs={'weight_decay': 1e-4, },
                # sim_budget=85, # how many simulations we choose randomly
                # epochs=600,
               )

# these are the settings that are taken from a pre-trained model
arch_settings = ['method', 'model', 'consider_params', 'priors', ]

filebase = '/tigress/lthiele/nuvoid_production'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

version = int(argv[1])
compression_hash = argv[2]
try :
    pretrained_ident = argv[3]
except IndexError :
    pretrained_ident = None

if pretrained_ident is not None :
    pretrained_fname = f'{filebase}/lfi_model_v{version}_{compression_hash}_{pretrained_ident}.sbi'
    print(f'Loading pretrained model from {pretrained_fname}')
    pretrained_settings, pretrained_posterior = load_posterior(pretrained_fname, device)
    for s in arch_settings :
        SETTINGS[s] = pretrained_settings[s]
    # by setting this, we ensure that the new file will have a different hash, and also useful for reference
    SETTINGS['pretrained_ident'] = pretrained_ident
else :
    pretrained_posterior = None

ident = hashlib.md5(f'{SETTINGS}'.encode('utf-8')).hexdigest()
print(f'ident={ident}')

data_fname = f'{filebase}/datavectors_trials.npz'
fiducials_fname = f'{filebase}/datavectors_fiducials_v0.npz'
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
    sim_idx = f['sim_idx']
data = np.einsum('ab,ib->ia', compression_matrix, data/normalization[None, :])


# restrict to prior
nbefore = len(params)
for k, (lo, hi) in SETTINGS['priors'].items() :
    param_idx = param_names.index(k)
    mask = (params[:, param_idx]>=lo) * (params[:, param_idx]<=hi)
    params = params[mask]
    data = data[mask]
    sim_idx = sim_idx[mask]
nafter = len(params)
print(f'Retaining {nafter/nbefore * 100} percent of samples after prior')

param_indices = [param_names.index(s) for s in SETTINGS['consider_params']]
params = params[:, param_indices]

if 'sim_budget' in SETTINGS :
    uniq_indices = np.unique(sim_idx)
    assert SETTINGS['sim_budget'] <= len(uniq_indices)
    rng = np.random.default_rng(137)
    use_indices = rng.choice(uniq_indices, replace=False, size=SETTINGS['sim_budget'])
    mask = np.array([idx in use_indices for idx in sim_idx], dtype=bool)
    data = data[mask]
    params = params[mask]
    sim_idx = sim_idx[mask]

if 'chisq_max' in SETTINGS :
    with np.load(fiducials_fname) as f :
        fid_mean = np.mean(f['data'], axis=0)
    fid_mean = compression_matrix @ (fid_mean/normalization)
    chisq = np.sum((data - fid_mean[None, :])**2, axis=-1) / data.shape[-1]
    mask = chisq < SETTINGS['chisq_max']
    print(f'After chisq_max cut, retaining {np.count_nonzero(mask)/len(mask)*100:.2f} percent of samples')
    data = data[mask]
    params = params[mask]
    sim_idx = sim_idx[mask]

# split off validation set by cosmology to get truly independent samples
#VAL_FRAC = 0.1
#rng = np.random.default_rng(137)
#uniq_indices = np.unique(sim_idx)
#val_sim_idx = rng.choice(uniq_indices, replace=False, size=int(VAL_FRAC * len(uniq_indices)))

# handpick 13 indices for validation, these are well-spaced in Mnu so they make a good validation set
val_sim_idx = [ 93, # 0.005258850
                21, # 0.052294624
                80, # 0.097084754
                 8, # 0.144120528
                67, # 0.188910657
               126, # 0.233700787
                73, # 0.284991010
                39, # 0.340535681
                98, # 0.385325810
                26, # 0.432361585
                85, # 0.477151714
               125, # 0.517687395
                34, # 0.560468721
              ]
validation_mask = np.array([idx in val_sim_idx for idx in sim_idx], dtype=bool)
validation_indices = validation_mask.nonzero()[0]

if 'noise' in SETTINGS :
    assert 'Mnu' in SETTINGS['consider_params']
    rng = np.random.default_rng(137)
    params[:, SETTINGS['consider_params'].index('Mnu')] += rng.normal(0, SETTINGS['noise'], size=len(params))

theta = torch.from_numpy(params.astype(np.float32)).to(device=device)
x = torch.from_numpy(data.astype(np.float32)).to(device=device)

inference = inference.append_simulations(theta=theta, x=x)
MAX_NUM_EPOCHS = SETTINGS['epochs'] if 'epochs' in SETTINGS else 200

if pretrained_posterior is not None :
    # if pretrained, set the neural net
    inference._neural_net = pretrained_posterior.potential_fn.ratio_estimator
density_estimator = inference.train(max_num_epochs=MAX_NUM_EPOCHS,
                                    stop_after_epochs=MAX_NUM_EPOCHS // 5, # the default, 20, is pretty short
                                    training_batch_size=SETTINGS['bs'] if 'bs' in SETTINGS else 50,
                                    learning_rate=SETTINGS['lr'] if 'lr' in SETTINGS else 5e-4,
                                    scheduler_kwargs=None if 'one_cycle' not in SETTINGS or not SETTINGS['one_cycle']
                                                     else {'max_lr': SETTINGS['lr'] if 'lr' in SETTINGS else 5e-4,
                                                           'total_steps': MAX_NUM_EPOCHS+3,
                                                           'final_div_factor': 50, # estimated empirically as better
                                                           'verbose': True},
                                    optimizer_kwargs={} if 'optimizer_kwargs' not in SETTINGS
                                                     else SETTINGS['optimizer_kwargs'],
                                    validation_indices=validation_indices,
                                   )
prior = sbi_utils.BoxUniform(low=torch.Tensor([SETTINGS['priors'][s][0] for s in SETTINGS['consider_params']]),
                             high=torch.Tensor([SETTINGS['priors'][s][1] for s in SETTINGS['consider_params']]),
                             device=device)
posterior = inference.build_posterior(density_estimator, prior=prior)

# useful to have this as a record, perhaps for ensembling
SETTINGS['best_val_log_prob'] = inference._best_val_log_prob

with open(model_fname, 'w') as f :
    f.write(f'{SETTINGS}\n')
with open(model_fname, 'ab') as f :
    pickle.dump(posterior, f)
