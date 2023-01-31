import numpy as np

from collect_datavectors import get_datavec_from_fnames

database = '/tigress/lthiele/boss_dr12'
filebase = '/tigress/lthiele/nuvoid_production'

voids_fname = f'{database}/voids/sample_test/sky_positions_central_test.out'
vgplk_fname = f'{database}/CMASS_North_vgplk.npz'
plk_fname = f'{database}/CMASS_North_plk.npz'

d = get_datavec_from_fnames(voids_fname, vgplk_fname, plk_fname)
np.savetxt(f'{filebase}/datavector_CMASS_North.dat', d)
