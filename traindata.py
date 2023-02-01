import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from read_txt import read_txt

filebase = '/tigress/lthiele/nuvoid_production'

class TrainData :

    # which parameters we regard as independent
    use_params = [
                  'Obh2',
                  'Och2',
                  'theta',
                  'logA',
                  'ns',
                  #'Om',
                  #'Ob',
                  #'h',
                  #'sigma8',
                  #'S8',
                  #'1e9As',
                  #'On',
                  #'Oc',
                  'Mnu',
                  # the following probably shouldn't be touched
                  'hod_transfP1', 'hod_abias', 'hod_log_Mmin', 'hod_sigma_logM', 'hod_log_M0', 'hod_log_M1',
                  'hod_alpha', 'hod_transf_eta_cen', 'hod_transf_eta_sat', 'hod_mu_Mmin', 'hod_mu_M1',
                 ]

    validation_frac = 0.2

    def __init__ (self, version, compression_hash, device, batch_size, use_avg) :
        
        data_fname = f'{filebase}/{"avg_" if use_avg else ""}datavectors_trials.npz'
        compress_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'
        fiducials_fname = f'{filebase}/datavectors_fiducials_v{version}.npz'

        # these are the settings used in compression
        self.consider_params = read_txt(compress_fname, 'consider_params:', pyobj=True)
        self.cut_kwargs = read_txt(compress_fname, 'cut_kwargs:', pyobj=True)

        # load the data
        with np.load(data_fname) as f :
            sim_idx = f['sim_idx']
            data = f['data'].astype(np.float64)
            param_names = list(f['param_names'])
            params = f['params']
            if use_avg :
                self.nsims = f['nsims']
            else :
                self.nsims = np.ones(len(data))

        # compress the data
        normalization = read_txt(compress_fname, 'normalization:')
        compression_matrix = read_txt(compress_fname, 'compression matrix:')
        self.data = np.einsum('ab,ib->ia', compression_matrix, data/normalization[None, :])

        # filter for independent parameters
        param_indices = [param_names.index(s) for s in TrainData.use_params]
        self.params = params[:, param_indices]

        # load the fiducials and compute chisq with respect to them
        with np.load(fiducials_fname) as f :
            fid_mean = np.mean(f['data'], axis=0)
        fid_mean = np.einsum('ab,b->a', compression_matrix, fid_mean/normalization)
        self.chisq = np.mean((self.data - fid_mean[None, :])**2, axis=-1)

        # split into training and validation set
        validation_mask = self._get_validation_mask(sim_idx)
        self.train_params = self.params[~validation_mask]
        self.validation_params = self.params[validation_mask]
        self.train_y = self.data[~validation_mask]
        self.validation_y = self.data[validation_mask]
        self.train_chisq = self.chisq[~validation_mask]
        self.validation_chisq = self.chisq[validation_mask]
        self.train_nsims = self.nsims[~validation_mask]
        self.validation_nsims = self.nsims[validation_mask]

        # normalize the inputs
        self.norm_avg = np.mean(self.train_params, axis=0)
        self.norm_std = np.std(self.train_params, axis=0)
        self.train_x = (self.train_params - self.norm_avg[None, :]) / self.norm_std[None, :]
        self.validation_x = (self.validation_params - self.norm_avg[None, :]) / self.norm_std[None, :]

        # move to torch
        self.train_x = torch.from_numpy(self.train_x.astype(np.float32)).to(device=device)
        self.train_y = torch.from_numpy(self.train_y.astype(np.float32)).to(device=device)
        self.train_chisq = torch.from_numpy(self.train_chisq.astype(np.float32)).to(device=device)
        self.train_nsims = torch.from_numpy(self.train_nsims.astype(np.float32)).to(device=device)
        self.validation_x = torch.from_numpy(self.validation_x.astype(np.float32)).to(device=device)
        self.validation_y = torch.from_numpy(self.validation_y.astype(np.float32)).to(device=device)
        self.validation_chisq = torch.from_numpy(self.validation_chisq.astype(np.float32)).to(device=device)
        self.validation_nsims = torch.from_numpy(self.validation_nsims.astype(np.float32)).to(device=device)

        # construct sets and loaders
        self.train_set = TensorDataset(self.train_x, self.train_y,
                                       self.train_chisq, self.train_nsims)
        self.validation_set = TensorDataset(self.validation_x, self.validation_y,
                                            self.validation_chisq, self.validation_nsims)
        self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size=batch_size)
        self.validation_loader = DataLoader(self.validation_set, shuffle=False, batch_size=512)


    def _get_validation_mask (self, sim_idx) :
        """ splits into training and validation set, such that there is no simulation overlap """
        rng = np.random.default_rng(137)
        uniq_indices = np.unique(sim_idx)
        validation_indices = rng.choice(uniq_indices, replace=False,
                                        size=int(TrainData.validation_frac * len(uniq_indices)))
        return np.array([idx in validation_indices for idx in sim_idx], dtype=bool)
