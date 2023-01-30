from sys import argv

import numpy as np
from sklearn.linear_model import LinearRegression

from cut import Cut

filebase = '/tigress/lthiele/nuvoid_production'

class LinRegress :

    # these are the independent parameters we use, different choices may lead
    # to better performance
    use_params = [
                  #'Om',
                  #'Ob',
                  #'h',
                  'ns',
                  #'sigma8',
                  #'S8',
                  'Mnu',
                  #'1e9As',
                  #'On',
                  #'Oc',
                  'Obh2',
                  'Och2',
                  'theta',
                  'logA',
                  # the following probably shouldn't be touched
                  'hod_transfP1', 'hod_abias', 'hod_log_Mmin', 'hod_sigma_logM', 'hod_log_M0', 'hod_log_M1',
                  'hod_alpha', 'hod_transf_eta_cen', 'hod_transf_eta_sat', 'hod_mu_Mmin', 'hod_mu_M1',
                 ]

    def __init__ (self, version, cut=None) :

        derivatives_fname = f'{filebase}/avg_datavectors_derivatives_v{version}.npz'
        fiducials_fname = f'{filebase}/datavectors_fiducials_v{version}.npz'

        with np.load(derivatives_fname) as f :
            derivatives_data = f['data']
            derivatives_params = f['params']
            derivatives_nsims = f['nsims'].astype(float)
            param_names = list(f['param_names'])
        with np.load(fiducials_fname) as f :
            fiducials_data = f['data']

        self.n_samples = derivatives_data.shape[0]

        param_indices = [param_names.index(s) for s in LinRegress.use_params]
        self.x = derivatives_params[:, param_indices]
        self.prior_ranges = [(np.min(x), np.max(x)) for x in self.x.T]

        if cut is not None :
            derivatives_data = cut.cut_vec(derivatives_data)
            fiducials_data = cut.cut_vec(fiducials_data)

        self.fid_mean = np.mean(fiducials_data, axis=0)

        # divide out the fiducial mean for better interpretability
        y = derivatives_data / self.fid_mean[None, :]

        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.x, y, sample_weight=derivatives_nsims)

        self.dm_dphi = self.model.coef_.T # shape [params, data]
        
        delta = self.model.predict(self.x) - y
        self.cov = np.cov(fiducials_data/self.fid_mean[None, :], rowvar=False)
        covinv = np.linalg.inv(self.cov)
        self.sample_chisq = np.einsum('s,sa,ab,sb->s', derivatives_nsims, delta, covinv, delta) / delta.shape[1]
        self.chisq = np.mean(self.sample_chisq)

    def __call__ (self, x) :
        return self.model.predict(x)


if __name__ == '__main__' :
    from matplotlib import pyplot as plt
    version = int(argv[1])
    cut = Cut()
    linregress = LinRegress(version, cut)
    print(f'Using {linregress.n_samples} samples')
    print(f'chisq/dof={linregress.chisq}')

    fig, ax = plt.subplots(nrows=2, figsize=(10,10))
    ax[0].semilogy(np.fabs(linregress.fid_mean))
    for name, dm, p in zip(LinRegress.use_params, linregress.dm_dphi, linregress.prior_ranges) :
        y = dm/np.sqrt(np.diagonal(linregress.cov)) * (p[1] - p[0])
        ax[1].plot(y, label=name, linestyle='dashed' if 'hod' in name else 'solid')
    ax[1].set_yscale('symlog')
    ax[1].legend(ncol=3)
    fig.savefig('derivatives1.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(10,20))
    ax = ax.flatten()
    for ii, name in enumerate(LinRegress.use_params) :
        ax[ii].scatter(linregress.x[:, ii], linregress.sample_chisq)
        ax[ii].set_xlabel(name)
    fig.savefig('derivatives2.pdf', bbox_inches='tight')