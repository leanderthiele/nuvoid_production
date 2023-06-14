import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from _plot_style import *

version = 0
filebase = '/tigress/lthiele/nuvoid_production'
fiducials_fname = f'{filebase}/datavectors_fiducials_v{version}.npz'

with np.load(fiducials_fname) as f :
    data = f['data']

# extract the VSF
data = data[:, :64]
avg = np.mean(data, axis=0)
cov = np.cov(data, rowvar=False)
cov_rescaled = cov / np.sqrt(avg[:,None] * avg[None,:])

fig, ax = plt.subplots(ncols=2, figsize=(5,3))
ax_cov = ax[0]
ax_asc = ax[1]

# check the covariance matrix versus Poissonian expectation
im = ax_cov.matshow(cov_rescaled, extent=(0,1,1,0), cmap='seismic', vmin=-1.2, vmax=1.2)
divider = make_axes_locatable(ax_cov)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='horizontal', label='$C_{ij}/\sqrt{\mu_i \mu_j}$')

ax_cov.set_xticks([0.5, 1.0])
ax_cov.set_xticklabels(['30 Mpc/h', '80'])
ax_cov.set_yticks([0.0, 0.5])
ax_cov.set_yticklabels(['30', '80'])
ax_cov.text(0.25,0.25,'z<0.53',color='black',transform=ax_cov.transData,va='center',ha='center')
ax_cov.text(0.75,0.75,'z>0.53',color='black',transform=ax_cov.transData,va='center',ha='center')

# check the distribution of Anscombe transformed counts
asc = 2 * np.sqrt(data+3/8) - 2 * np.sqrt(avg+3/8) + 1/(4*np.sqrt(avg))
asc = asc.flatten()
lim = (-4, 4)
ax_asc.hist(asc, range=lim, bins=32, density=True, histtype='step')
x = np.linspace(*lim, num=100)
gaussian = 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
ax_asc.plot(x, gaussian, color=black, linestyle='dashed')
ax_asc.set_xlim(*lim)
ax_asc.set_yticks([])
ax_asc.set_xlabel('Anscombe transformed VSF')

savefig(fig, 'poisson')
