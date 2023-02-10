import numpy as np
from matplotlib import pyplot as plt

from read_txt import read_txt
from cut import Cut

consider_hashes = {
                   '558e99a0b3ea41d1976b469af5c8a461': 'all, kmax=0.1',
                   '1b33d8416f5a164be6f8850c11214029': 'all, kmax=0.15',
                   'f0a5be2c040b44c8269f5c8b9a48a79f': 'plk, kmax=0.15',
                   'f838d51a7c13abe194429f6e87d8cb12': 'vsf+plk, kmax=0.15',
                   '76c30e9ea67c1464374e21ea832cbbb2': 'vsf',
                  }

have_params = ['Mnu', 'hod_log_Mmin', ]

version = 0

filebase = '/tigress/lthiele/nuvoid_production'

target_data = np.loadtxt(f'{filebase}/datavector_CMASS_North.dat')

with np.load(f'{filebase}/avg_datavectors_trials.npz') as f :
    trial_data = f['data']
    trial_params = f['params']
    param_names = list(f['param_names'])

with np.load(f'{filebase}/datavectors_fiducials_v{version}.npz') as f :
    cov = np.cov(f['data'], rowvar=False)
    fid_mean = np.mean(f['data'], axis=0)

xindices = np.arange(len(target_data))

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

fig, ax = plt.subplots(nrows=1+len(consider_hashes), figsize=(20,20),
                       gridspec_kw=dict(hspace=0))
ax_datavec = ax[0]
ax_residuals = ax[1:]

ax_datavec.plot(xindices, target_data, linestyle='none', marker='o', label='CMASS NGC')
ax_datavec.plot(xindices, fid_mean, label=f'fiducial v{version}')

for compression_hash, a in zip(consider_hashes.keys(), ax_residuals) :

    compression_file = f'{filebase}/compression_v{version}_{compression_hash}.dat'
    label = f'{compression_hash[:4]} {consider_hashes[compression_hash]}'

    cut_kwargs = read_txt(compression_file, 'cut_kwargs:', pyobj=True)
    cut = Cut(**cut_kwargs)

    target_data_cut = cut.cut_vec(target_data)
    trial_data_cut = cut.cut_vec(trial_data)
    covinv_cut = np.linalg.inv(cut.cut_mat(cov))
    delta = trial_data_cut - target_data_cut[None, :]
    chisq_cut = np.einsum('ia,ab,ib->i', delta, covinv_cut, delta)
    min_idx_cut = np.argmin(chisq_cut)

    normalization = read_txt(compression_file, 'normalization:')
    compression_matrix = read_txt(compression_file, 'compression matrix:')
    target_data_cmpr = compression_matrix @ (target_data/normalization)
    trial_data_cmpr = np.einsum('ab,ib->ia', compression_matrix, trial_data/normalization[None, :])
    chisq_cmpr = np.sum((trial_data_cmpr - target_data_cmpr[None, :])**2, axis=-1)
    min_idx_cmpr = np.argmin(chisq_cmpr)

    trial_data_bf_cut = trial_data[min_idx_cut]
    trial_data_bf_cmpr = trial_data[min_idx_cmpr]

    chisq_red_bf_cut = chisq_cut[min_idx_cut] / len(target_data_cut)
    chisq_red_bf_cmpr = chisq_cmpr[min_idx_cmpr] / len(target_data_cmpr)

    delta_bf_cut = (trial_data_bf_cut - target_data) / np.sqrt(np.diagonal(cov))
    delta_bf_cmpr = (trial_data_bf_cmpr - target_data) / np.sqrt(np.diagonal(cov))

    # indication of best-fit parameters
    param_str_cut = ' '.join(f'{s}={trial_params[min_idx_cut, param_names.index(s)]:.2f}' for s in have_params)
    param_str_cmpr = ' '.join(f'{s}={trial_params[min_idx_cmpr, param_names.index(s)]:.2f}' for s in have_params)

    # first do the used data
    l = a.plot(xindices[cut.mask], delta_bf_cut[cut.mask], linestyle='none', marker='o',
                          label=f'{label} cut $\\chi^2_{{\\sf red}}$={chisq_red_bf_cut:.2f} {param_str_cut}')
    l = l[0]
    a.plot(xindices[cut.mask], delta_bf_cmpr[cut.mask], linestyle='none', marker='s',
                      color=l.get_color(),
                      label=f'{label} compressed $\\chi^2_{{\\sf red}}$={chisq_red_bf_cmpr:.2f} {param_str_cmpr}')

    # now the unused
    a.plot(xindices[~cut.mask], delta_bf_cut[~cut.mask], linestyle='none', marker='o',
                      markerfacecolor='none', color=l.get_color())
    a.plot(xindices[~cut.mask], delta_bf_cmpr[~cut.mask], linestyle='none', marker='s',
                      markerfacecolor='none', color=l.get_color())

ax_datavec.set_yscale('symlog')
ax_datavec.legend()

linthresh = 4
for a in ax_residuals :
    #a.set_ylim(-10,10)
    a.set_yscale('symlog', linthresh=linthresh)
    a.set_yticks([-10, -4, -3, -2, -1, 0, 1, 2, 3, 4, 10])
    a.axhline(0, color='grey', linestyle='dashed')
    a.axhline(+linthresh, color='grey', linestyle='dotted')
    a.axhline(-linthresh, color='grey', linestyle='dotted')
    a.legend(loc='upper left', ncol=2)
    a.set_ylabel('$\Delta/\sigma$')

for a in ax :
    a.set_xticks([])
    for vline in vlines :
        a.axvline(vline, color='grey')

for a in ax :
    ymin, _ = a.get_ylim()
    for ii, desc in enumerate(part_desc) :
        a.text(0.5*(vlines[ii]+vlines[ii+1]), ymin, desc,
               transform=a.transData, va='bottom', ha='center')

fig.savefig('bestfit.pdf', bbox_inches='tight')
