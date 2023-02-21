from glob import glob

import numpy as np
from matplotlib import pyplot as plt

rng = np.random.default_rng(137)

zmin, zmax = 0.42, 0.70

nrows, ncols = 3, 3

dec0 = 20.0
delta_dec = 1.0

z0 = 0.6
delta_z = 0.01

fig_kwargs = dict(nrows=nrows, ncols=ncols, figsize=(10,10),
                  gridspec_kw=dict(hspace=0, wspace=0))
scatter_kwargs = dict(s=0.1)

# slices of constant dec
fig_dec, ax_dec = plt.subplots(**fig_kwargs)
ax_dec_f = ax_dec.flatten()

# slices of constant z
fig_z, ax_z = plt.subplots(**fig_kwargs)
ax_z_f = ax_z.flatten()

# find the available fiducial lightcones
fid_lc_files = glob('/scratch/gpfs/lthiele/nuvoid_production/cosmo_fiducial_*/lightcones/c2c93dbc97d64a7c20a043121f7d23d8/lightcone_*.bin')
print(f'Found {len(fid_lc_files)} fiducial lightcones')

# choose the lightcones to use
rnd_indices = rng.choice(len(fid_lc_files), size=len(ax_dec_f)-1, replace=False)
lc_files = [fid_lc_files[ii] for ii in rnd_indices]

# where to insert the real universe
real_index = rng.integers(len(ax_dec_f))
print(f'real_index={real_index}')

lc_files.insert(real_index, '/tigress/lthiele/boss_dr12/galaxy_DR12v5_CMASS_North.bin')

for lc_file, a_dec, a_z in zip(lc_files, ax_dec_f, ax_z_f) :
    ra, dec, z = np.fromfile(lc_file).reshape(3, -1)

    mask = (z>zmin) *(z<zmax)
    ra = ra[mask]
    dec = dec[mask]
    z = z[mask]

    dec_mask = (dec>(dec0-0.5*delta_dec)) * (dec<(dec0+0.5*delta_dec))
    z_mask = (z>(z0-0.5*delta_z)) * (z<(z0+0.5*delta_z))

    a_dec.scatter(ra[dec_mask], z[dec_mask], **scatter_kwargs)
    a_z.scatter(ra[z_mask], dec[z_mask], **scatter_kwargs)

for yax, ax in zip(['z', 'DEC', ], [ax_dec, ax_z, ]) :
    for row, ax_row in enumerate(ax) :
        for col, a in enumerate(ax_row) :
            if row != nrows - 1 :
                a.set_xticks([])
            else :
                a.set_xlabel('RA')
            if col != 0 :
                a.set_yticks([])
            else :
                a.set_ylabel(yax)

fig_dec.savefig('_plot_spot_the_universe_decslice.pdf', bbox_inches='tight')
fig_z.savefig('_plot_spot_the_universe_zslice.pdf', bbox_inches='tight')

for a in [ax_dec_f, ax_z_f, ] :
    for pos in ['top', 'bottom', 'right', 'left', ] :
        s = a[real_index].spines[pos]
        plt.setp(s.set(edgecolor='red', linewidth=2)

fig_dec.savefig('_plot_spot_the_universe_revealed_decslice.pdf', bbox_inches='tight')
fig_z.savefig('_plot_spot_the_universe_revealed_zslice.pdf', bbox_inches='tight')

