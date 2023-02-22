from glob import glob

import numpy as np
from matplotlib import pyplot as plt

plt.style.use('dark_background')

rng = np.random.default_rng(42)

zmin, zmax = 0.42, 0.70

nrows, ncols = 3, 3

dec0 = 20.0
delta_dec = 1.0

z0 = 0.6
delta_z = 0.005

fig_kwargs = dict(nrows=nrows, ncols=ncols, figsize=(10,10),
                  gridspec_kw=dict(hspace=0.1, wspace=0.1))
plot_kwargs = dict(linestyle='none', marker='o', markersize=0.1)
save_kwargs = dict(bbox_inches='tight', transparent=True)
fmt = 'png'

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

lc_files.insert(real_index, '/tigress/lthiele/boss_dr12/galaxy_DR12v5_CMASS_North.bin')

for lc_file, a_dec, a_z in zip(lc_files, ax_dec_f, ax_z_f) :
    ra, dec, z = np.fromfile(lc_file).reshape(3, -1)

    mask = (z>zmin) *(z<zmax)
    ra = ra[mask]
    dec = dec[mask]
    z = z[mask]

    dec_mask = (dec>(dec0-0.5*delta_dec)) * (dec<(dec0+0.5*delta_dec))
    z_mask = (z>(z0-0.5*delta_z)) * (z<(z0+0.5*delta_z))

    a_dec.plot(ra[dec_mask], z[dec_mask], **plot_kwargs)
    a_z.plot(ra[z_mask], dec[z_mask], **plot_kwargs)

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
            for pos in ['top', 'bottom', 'right', 'left', ] :
                a.spines[pos].set(edgecolor='none')

for s, f in zip(['dec', 'z', ], [fig_dec, fig_z, ]) :
    f.savefig(f'_plot_spot_the_universe_{s}slice.{fmt}', **save_kwargs)

for a in [ax_dec_f, ax_z_f, ] :
    for pos in ['top', 'bottom', 'right', 'left', ] :
        s = a[real_index].spines[pos]
        s.set(edgecolor='green', linewidth=4)

for s, f in zip(['dec', 'z', ], [fig_dec, fig_z, ]) :
    f.savefig(f'_plot_spot_the_universe_revealed_{s}slice.{fmt}', **save_kwargs)
