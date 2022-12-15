import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.stats import scott_bin_width
import nbodykit.lab as NBL

class PLKCalc :
    
    # some settings
    h = 0.71 # found this somewhere in the VIDE source, shouldn't matter
    Om = 0.3439 # this is what we used for the VIDE runs (first emulator)
    P0 = 1e4

    Nmesh = 360
    kmax = 0.2
    dk = 0.005
    poles = [0, 2, 4]

    # these were used when constructing the lightcones
    zmin = 0.42
    zmax = 0.70
    
    rand_file = '/tigress/lthiele/boss_dr12/random_DR12v5_CMASS_North_downsampled6188060.npz'
    # from Monte Carlo, valid for CMASS North
    fsky = 0.16795440

    
    def __init__(self) :
        self.cosmo = NBL.cosmology.Cosmology(h=PLKCalc.h).match(Omega0_m=PLKCalc.Om)
        with np.load(PLKCalc.rand_file) as f :
            ra_rand, dec_rand, z_rand = [f[s] for s in ['RA', 'DEC', 'Z']]
        zselect = (z_rand>PLKCalc.zmin) * (z_rand<PLKCalc.zmax)
        ra_rand = ra_rand[zselect]
        dec_rand = dec_rand[zselect]
        self.z_rand = z_rand[zselect] # need to store for n_bar
        self.pos_rand = NBL.transform.SkyToCartesian(ra_rand, dec_rand, self.z_rand, cosmo=self.cosmo)


    def compute_from_arrays(self, ra_gals, dec_gals, z_gals) :
        zselect = (z_gals>PLKCalc.zmin) * (z_gals<PLKCalc.zmax)
        ra_gals = ra_gals[zselect]
        dec_gals = dec_gals[zselect]
        z_gals = z_gals[zselect]

        ng_of_z = self.nz(z_gals)
        nbar_gals = ng_of_z(z_gals)
        nbar_rand = ng_of_z(self.z_rand)

        fkp_gals = 1 / (1 + nbar_gals * PLKCalc.P0)
        fkp_rand = 1 / (1 + nbar_rand * PLKCalc.P0)
        
        pos_gals = NBL.transform.SkyToCartesian(ra_gals, dec_gals, z_gals, cosmo=self.cosmo)


        cat_gals = NBL.ArrayCatalog({'Position': pos_gals, 'NZ': nbar_gals,
                                     'WEIGHT': np.ones(len(z_gals)), 'WEIGHT_FKP': fkp_gals})
        cat_rand = NBL.ArrayCatalog({'Position': self.pos_rand, 'NZ': nbar_rand,
                                     'WEIGHT': np.ones(len(self.z_rand)), 'WEIGHT_FKP': fkp_rand})

        cat_fkp = NBL.FKPCatalog(cat_gals, cat_rand)
        mesh = cat_fkp.to_mesh(Nmesh=PLKCalc.Nmesh, nbar='NZ', fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT',
                               window='tsc')

        res = NBL.ConvolvedFFTPower(mesh, poles=PLKCalc.poles, dk=PLKCalc.dk, kmax=PLKCalc.kmax)

        return dict(k=res.poles['k'],
                    **{f'p{ell}k': res.poles[f'power_{ell}'].real - (res.attrs['shotnoise'] if ell==0 else 0)
                       for ell in PLKCalc.poles}
                   )


    def compute_from_fname(self, fname) :
        # expects a binary file in the order RA, DEC, Z,
        # 8 byte doubles
        return self.compute_from_arrays(*np.fromfile(fname).reshape(3, -1))


    def nz(self, z) :
        _, edges = scott_bin_width(z, return_bins=True)
        dig = np.searchsorted(edges, z, 'right')
        N = np.bincount(dig, minlength=len(edges)+1)[1:-1]
        R_hi = self.cosmo.comoving_distance(edges[1:])
        R_lo = self.cosmo.comoving_distance(edges[:-1])
        dV = 4*np.pi/3 * (R_hi**3 - R_lo**3) * PLKCalc.fsky
        return InterpolatedUnivariateSpline(0.5*(edges[1:]+edges[:-1]), N/dV, ext='const')
