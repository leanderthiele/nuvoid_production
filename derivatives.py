from sys import argv

import numpy as np
from sklearn.linear_model import LinearRegression

from cut import Cut

filebase = '/tigress/lthiele/nuvoid_production'

class LinRegress :

    # these are the independent parameters we use, different choices may lead
    # to better performance
    # The CMB parameterization works well, keep it in the order of our prior
    # for convenience
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

    def __init__ (self, version, cut=None, deriv_fraction=None) :
        """ the cut argument is only useful for easy diagnostic output,
            we probably don't want to use it in production
        """

        derivatives_fname = f'{filebase}/avg_datavectors_derivatives_v{version}.npz'
        fiducials_fname = f'{filebase}/datavectors_fiducials_v{version}.npz'

        with np.load(derivatives_fname) as f :
            derivatives_data = f['data']
            derivatives_params = f['params']
            derivatives_nsims = f['nsims'].astype(float)
            param_names = list(f['param_names'])
        with np.load(fiducials_fname) as f :
            fiducials_data = f['data']

        if deriv_fraction is not None :
            rng = np.random.default_rng()
            indices = rng.choice(len(derivatives_data), int(deriv_fraction*len(derivatives_data)), replace=False)
            derivatives_data = derivatives_data[indices]
            derivatives_params = derivatives_params[indices]
            derivatives_nsims = derivatives_nsims[indices]

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
    linregress = LinRegress(version, cut=None)

    xindices = np.arange(len(cut.mask))
    vlines = [-0.5, ]
    part_desc = []
    for zbin in Cut.vsf_zbins :
        vlines.append(vlines[-1] + len(Cut.vsf_R))
        part_desc.append(f'vsf z{zbin}')
    for Rmin in Cut.vgplk_Rbins :
        for ell in Cut.vgplk_ell :
            vlines.append(vlines[-1] + len(Cut.vgplk_k))
            part_desc.append(f'$P_{ell}^{{vg}}$ $R_{{\\sf min}}={Rmin}$')
    for ell in Cut.plk_ell :
        vlines.append(vlines[-1] + len(Cut.plk_k))
        part_desc.append(f'$P_{ell}^{{gg}}$')

    print(f'Using {linregress.n_samples} samples')
    print(f'chisq/dof={linregress.chisq}')

    fig, ax = plt.subplots(nrows=2, figsize=(15,15))
    ax[0].semilogy(np.fabs(linregress.fid_mean))
    for name, dm, p in zip(LinRegress.use_params, linregress.dm_dphi, linregress.prior_ranges) :
        y = dm/np.sqrt(np.diagonal(linregress.cov)) * (p[1] - p[0])
        label = name if 'hod' not in name else name[4:]
        ax[1].plot(y, label=label, linestyle='dashed' if 'hod' in name else 'solid')
    ax[1].set_yscale('symlog')
    ax[1].legend(ncol=4)
    ymin, _ = ax[1].get_ylim()
    for vline in vlines :
        ax[1].axvline(vline, color='grey')
    for ii, desc in enumerate(part_desc) :
        ax[1].text(0.5*(vlines[ii]+vlines[ii+1]), ymin, desc,
                   transform=ax[1].transData, va='bottom', ha='center')
    fig.savefig('derivatives1.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(10,20))
    ax = ax.flatten()
    for ii, name in enumerate(LinRegress.use_params) :
        ax[ii].scatter(linregress.x[:, ii], linregress.sample_chisq)
        ax[ii].set_xlabel(name)
    fig.savefig('derivatives2.pdf', bbox_inches='tight')
