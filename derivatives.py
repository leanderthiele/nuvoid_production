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
            param_names = f['param_names']
        with np.load(fiducials_fname) as f :
            fiducials_data = f['data']

        param_indices = [param_names.index(s) for s in LinRegress.use_params]
        x = derivatives_params[:, param_indices]

        if cut is not None :
            derivatives_data = cut.cut_vec(derivatives_data)
            fiducials_data = cut.cut_vec(fiducials_data)

        self.fid_mean = np.mean(fiducials_data, axis=0)

        # divide out the fiducial mean for better interpretability
        y = derivatives_data / self.fid_mean[None, :]

        self.model = LinearRegression(fid_intercept=True)
        self.model.fit(x, y, sample_weight=derivatives_nsims)

        self.dm_dphi = self.model.coef_.T # shape [params, data]
        
        delta = self.model.predict(x) - y
        cov = np.cov(fiducials_data/self.fid_mean[None, :], rowvar=False)
        covinv = np.linalg.inv(cov)
        self.chisq = np.einsum('s,sa,ab,sb->', derivatives_nsims, delta, covinv, delta) / delta.shape[1]

    def __call__ (self, x) :
        return self.model.predict(x)


if __name__ == '__main__' :
    from matplotlib import pyplot as plt
    version = int(argv[1])
    cut = Cut()
    linregress = LinRegress(version, cut)
    print(f'chisq/dof={linregress.chisq}')

    fig, ax = plt.subplots(nrows=2)
    ax[0].semilogy(np.fabs(linregress.fid_mean))
    for name, dm in zip(LinRegress.use_params, linregress.dm_dphi) :
        ax[1].plot(dm, label=name)
    ax[1].legend()
    fig.savefig('derivatives.pdf', bbox_inches='tight')
