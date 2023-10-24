import numpy as np
from matplotlib import pyplot as plt

from astropy.cosmology import Planck18

from _plot_style import *

z0 = 0.6
deltaz_gal = 0.005
deltaz_voids = 0.005


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

filebase = '/tigress/lthiele/boss_dr12'
lc_file = f'{filebase}/galaxy_DR12v5_CMASS_North.bin'
voids_file = f'{filebase}/voids/sample_test/sky_positions_central_test.out'

ra_gal, dec_gal, z_gal = np.fromfile(lc_file).reshape(3, -1)
ra_voids, dec_voids, z_voids, R_voids = np.loadtxt(voids_file, usecols=(0, 1, 2, 3), unpack=True)

mask = (z_voids>z0-0.5*deltaz_voids) * (z_voids<z0+0.5*deltaz_voids) * (R_voids>30)
print(f'Have {np.count_nonzero(mask)} voids')
ra_voids = ra_voids[mask]
dec_voids = dec_voids[mask]
z_voids = z_voids[mask]
R_voids = R_voids[mask]

chi = Planck18.comoving_distance(z0).value * Planck18.h # Mpc / h
R_voids *= 180/np.pi/chi # in degrees now

mask = (z_gal>z0-0.5*deltaz_gal) * (z_gal<z0+0.5*deltaz_gal)
ra_gal = ra_gal[mask]
dec_gal = dec_gal[mask]
z_gal = z_gal[mask]

for pos in ['top', 'bottom', 'right', 'left', ] :
    ax.spines[pos].set(edgecolor='none')
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')


ax.plot(ra_gal, dec_gal, linestyle='none', marker='o', markersize=0.2)
savefig(fig, 'novoids')

for ra, dec, R in zip(ra_voids, dec_voids, R_voids) :
    c = plt.Circle((ra, dec), radius=R, fill=False, edgecolor='white')
    ax.add_patch(c)

savefig(fig, 'voids')
