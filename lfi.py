from sys import argv
import io
import os.path
import pickle
from multiprocessing import cpu_count
from datetime import datetime

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from sbi.inference import SNRE
from sbi import utils as sbi_utils

from read_txt import read_txt

filebase = '/tigress/lthiele/nuvoid_production'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Unpickler(pickle.Unpickler) :
    """ small utility to load cross-device """
    def find_class (self, module, name) :
        if module == 'torch.storage' and name == '_load_from_bytes' :
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else :
            return super().find_class(module, name)

class LFI :

    # only allow uniform priors for now
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
             }

    def __init__ (self, consider_params, version, compression_hash,
                        model='mlp', hidden_features=128) :
        
        self.consider_params = consider_params
        self.version = version
        self.compression_hash = compression_hash

        self.data_fname = f'{filebase}/datavectors_trials.npz'
        self.compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'
        self.model_fname = f'{filebase}/lfi_model_v{version}_{compression_hash}_{model}_{hidden_features}.sbi'
        self.tb_logdir = f'{filebase}/sbi_logs_v{version}_{compression_hash}/{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_{model}_{hidden_features}'

        self.normalization = read_txt(self.compress_fname, 'normalization:')
        self.compression_matrix = read_txt(self.compress_fname, 'compression matrix:')

        self.prior = sbi_utils.BoxUniform(low=torch.Tensor([LFI.priors[s][0] for s in self.consider_params]),
                                          high=torch.Tensor([LFI.priors[s][1] for s in self.consider_params]),
                                          device=device)

        self.make_model = sbi_utils.get_nn_models.classifier_nn(model=model,
                                                                z_score_theta='independent',
                                                                z_score_x='independent',
                                                                hidden_features=hidden_features)
        self.inference = SNRE(prior=self.prior, classifier=self.make_model, device=device, show_progress_bars=True,
                              summary_writer=SummaryWriter(log_dir=self.tb_logdir))

        if os.path.isfile(self.model_fname) :
            print(f'Found trained posterior in {self.model_fname}, loading')
            with open(self.model_fname, 'rb') as f :
                self.posterior = Unpickler(f).load()
                # fix some stuff
                self.posterior._device = device
                self.posterior.potential_fn.device = device
        else :
            print(f'Did not find trained posterior in {self.model_fname}')
            self.posterior = None


    def train (self) :

        with np.load(self.data_fname) as f :
            data = f['data'].astype(np.float64)
            param_names = list(f['param_names'])
            params = f['params']

        data = np.einsum('ab,ib->ia', self.compression_matrix, data/self.normalization[None, :])
        param_indices = [param_names.index(s) for s in self.consider_params]

        theta = torch.from_numpy(params[:, param_indices].astype(np.float32)).to(device=device)
        x = torch.from_numpy(data.astype(np.float32)).to(device=device)

        self.inference = self.inference.append_simulations(theta=theta, x=x)
        density_estimator = self.inference.train()
        self.posterior = self.inference.build_posterior(density_estimator)

        with open(self.model_fname, 'wb') as f :
            pickle.dump(self.posterior, f)

    
    def run_chain (self, observation) :
        """ returns the chain """
        
        if observation == 'cmass' :
            x = np.loadtxt(f'{filebase}/datavector_CMASS_North.dat')
            observation = self.compression_matrix @ (x/self.normalization)
        else :
            raise NotImplementedError

        # TODO there are a bunch of options here that we could explore
        chain = self.posterior.sample(sample_shape=(1000,), x=observation)
        return chain.cpu().numpy()


if __name__ == '__main__' :

    version = int(argv[1])
    compression_hash = argv[2]
    
    lfi = LFI(['Mnu', 'hod_log_Mmin', 'hod_mu_Mmin', ], version, compression_hash)
    if lfi.posterior is None :
        lfi.train()
    chain = lfi.run_chain('cmass')

    np.savez(f'{filebase}/lfi_chain_v{version}_{compression_hash}.npz',
             chain=chain, param_names=lfi.consider_params)
