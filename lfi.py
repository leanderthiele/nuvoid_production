from sys import argv

import numpy as np

import torch

from sbi.inference import SNRE
from sbi import utils as sbi_utils

from read_txt import read_txt

filebase = '/tigress/lthiele/nuvoid_production'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def __init__ (self, consider_params, version, compression_hash) :
        
        self.consider_params = consider_params
        self.version = version
        self.compression_hash = compression_hash

        data_fname = f'{filebase}/datavectors_trials.npz'
        compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'

        self.normalization = read_txt(compress_fname, 'normalization:')
        self.compression_matrix = read_txt(compress_fname, 'compression matrix:')

        with np.load(data_fname) as f :
            data = f['data'].astype(np.float64)
            param_names = list(f['param_names'])
            params = f['params']

        data = np.einsum('ab,ib->ia', self.compression_matrix, data/self.normalization[None, :])
        param_indices = [param_names.index(s) for s in self.consider_params]

        self.theta = torch.from_numpy(params[:, param_indices].astype(np.float32)).to(device=device)
        self.x = torch.from_numpy(data.astype(np.float32)).to(device=device)

        self.prior = sbi_utils.BoxUniform(low=torch.Tensor([LFI.priors[s][0] for s in self.consider_params]),
                                          high=torch.Tensor([LFI.priors[s][1] for s in self.consider_params]),
                                          device=device)

        self.make_model = sbi_utils.get_nn_models.classifier_nn(model='mlp',
                                                                z_score_theta='independent',
                                                                z_score_x='independent',
                                                                hidden_features=50)
        self.inference = SNRE(prior=self.prior, classifier=self.make_model, device=device, show_progress_bars=True)
        self.inference = self.inference.append_simulations(theta=self.theta, x=self.x)
        self.density_estimator = self.inference.train()
        self.posterior = self.inference.build_posterior(self.density_estimator)

    
    def __call__ (self, observation) :
        """ returns the chain """
        
        if observation == 'cmass' :
            x = np.loadtxt(f'{filebase}/datavector_CMASS_North.dat')
            observation = self.compression_matrix @ (x/self.normalization)
        else :
            raise NotImplementedError

        # TODO there are a bunch of options here that we could explore
        chain = self.posterior.sample(sample_shape=(2000,), x=observation)
        return chain.cpu().numpy()


if __name__ == '__main__' :

    version = int(argv[1])
    compression_hash = argv[2]
    
    lfi = LFI(['Mnu', 'hod_log_Mmin', 'hod_mu_Mmin', ], version, compression_hash)
    chain = lfi('cmass')

    np.savez(f'{filebase}/lfi_chain_v{version}_{compression_hash}.npz',
             chain=chain, param_names=lfi.consider_params)
