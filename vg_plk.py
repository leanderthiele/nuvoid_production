import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.stats import scott_bin_width
import nbodykit.lab as NBL
from pypower import MeshFFTPower, CatalogMesh
from sklearn.neighbors import KernelDensity

class PLKCalc :
    
    # some settings
    h = 0.71 # found this somewhere in the VIDE source, shouldn't matter
    Om = 0.3439 # this is what we used for the VIDE runs (first emulator)
    P0_gals = 1e4
    P0_voids = 1e3 # I got this value by looking at the vv line in Fig. 1 (right) of https://arxiv.org/pdf/1307.2571.pdf,
                   # which may be using a lot more voids. Probably worth playing with
                   # TODO

    Nmesh = 360 # TODO is this really enough???
    kmax = 0.2
    dk = 0.005
    poles = [0, 2, 4]

    # these were used when constructing the lightcones
    zmin = 0.42
    zmax = 0.70
    
    gals_rand_file = '/tigress/lthiele/boss_dr12/random_DR12v5_CMASS_North_downsampled6188060.npz'
    # from Monte Carlo, valid for CMASS North
    fsky = 0.16795440

    # void Rmin, Rmax values
    Rmin = [30, 40, 50, 60, ] # needs to be increasing
    Rmax = 80

    # how many random voids we use
    N_rand_voids = 200000

    # this file contains voids from our simulations, can be used to construct a randoms catalog
    # that respects the mask
    voids_file = '/home/lthiele/nuvoid_production/collected_voids.npz'


    def __init__(self, comm) :
        self.comm = comm
        self.cosmo = NBL.cosmology.Cosmology(h=PLKCalc.h).match(Omega0_m=PLKCalc.Om)
        with np.load(PLKCalc.gals_rand_file) as f :
            ra_rand, dec_rand, z_rand = [f[s] for s in ['RA', 'DEC', 'Z']]

        # get data for our rank
        lo, hi = self.interval(len(z_rand))
        ra_rand = ra_rand[lo:hi]
        dec_rand = dec_rand[lo:hi]
        z_rand = z_rand[lo:hi]

        zselect = (z_rand>PLKCalc.zmin) * (z_rand<PLKCalc.zmax)
        ra_rand = ra_rand[zselect]
        dec_rand = dec_rand[zselect]
        self.z_rand_gals = z_rand[zselect] # need to store for n_bar

        # pypower expects numpy, not dask arrays, this is why we need to compute()
        self.pos_rand_gals = NBL.transform.SkyToCartesian(ra_rand, dec_rand, self.z_rand_gals,
                                                          cosmo=self.cosmo).compute()

        # load the voids for randoms
        with np.load(PLKCalc.voids_file) as f :
            ra_voids, dec_voids, r_voids = [f[s] for s in ['RA', 'DEC', 'R']]
        lo, hi = self.interval(len(ra_voids))
        ra_voids = ra_voids[lo:hi]
        dec_voids = dec_voids[lo:hi]
        r_voids = r_voids[lo:hi]
        select = r_voids < PLKCalc.Rmax
        self.ra_dec_collected_voids = np.stack([ra_voids[select], dec_voids[select]], axis=1)
        self.r_collected_voids = r_voids[select]

        # to construct the random voids
        self.rng = np.random.default_rng(123456+self.comm.rank)

    
    def compute_from_arrays(self, ra_gals, dec_gals, z_gals,
                                  ra_voids, dec_voids, z_voids,
                                  R_voids) :

        # first put the galaxies on a mesh
        zselect = (z_gals>PLKCalc.zmin) * (z_gals<PLKCalc.zmax)
        ra_gals = ra_gals[zselect]
        dec_gals = dec_gals[zselect]

        ng_of_z = self.nz(z_gals)

        # get data for our rank
        lo, hi = self.interval(len(z_gals))
        ra_gals = ra_gals[lo:hi]
        dec_gals = dec_gals[lo:hi]
        z_gals = z_gals[lo:hi]

        nbar_gals = ng_of_z(z_gals)
        nbar_rand_gals = ng_of_z(self.z_rand_gals)

        fkp_gals = 1 / (1 + nbar_gals * PLKCalc.P0_gals)
        fkp_rand_gals = 1 / (1 + nbar_rand_gals * PLKCalc.P0_gals)

        pos_gals = NBL.transform.SkyToCartesian(ra_gals, dec_gals, z_gals,
                                                cosmo=self.cosmo).compute()
        mesh_gals = CatalogMesh(data_positions=pos_gals, data_weights=fkp_gals,
                                randoms_positions=self.pos_rand_gals, randoms_weights=fkp_rand_gals,
                                position_type='pos',
                                nmesh=PLKCalc.Nmesh,
                                mpicomm=self.comm).to_mesh(field='fkp')

        results = []
        for rmin in PLKCalc.Rmin :
            select = (R_voids > rmin) * (R_voids < PLKCalc.Rmax)
            ra_voids = ra_voids[select]
            dec_voids = dec_voids[select]
            z_voids = z_voids[select]
            R_voids = R_voids[select]
            select = self.r_collected_voids > rmin
            self.ra_dec_collected_voids = self.ra_dec_collected_voids[select]
            self.r_collected_voids = self.r_collected_voids[select]

            nv_of_z = self.nz(z_voids)
            ra_rand_voids, dec_rand_voids, z_rand_voids = self.rand_voids(z_voids)

            nbar_voids = nv_of_z(z_voids)
            nbar_rand_voids = nv_of_z(z_rand_voids)

            fkp_voids = 1 / (1 + nbar_voids * PLKCalc.P0_voids)
            fkp_rand_voids = 1 / (1 + nbar_rand_voids * PLKCalc.P0_voids)

            pos_voids = NBL.transform.SkyToCartesian(ra_voids, dec_voids, z_voids,
                                                     cosmo=self.cosmo).compute()
            pos_rand_voids = NBL.transform.SkyToCartesian(ra_rand_voids, dec_rand_voids, z_rand_voids,
                                                          cosmo=self.cosmo).compute()
            mesh_voids = CatalogMesh(data_positions=pos_voids, data_weights=fkp_voids,
                                     randoms_positions=pos_rand_voids, randoms_weight=fkp_rand_voids,
                                     position_type='pos',
                                     nmesh=PLKCalc.Nmesh,
                                     mpicomm=self.comm).to_mesh(field='fkp')

            p = MeshFFTPower(mesh_gals, mesh_voids, ells=[0, ], edges=np.linspace(0, 0.5, num=50))

            results.append({'k': p.k, 'kavg': p.kavg, 'Plk': p.power})

        return results


    def compute_from_fnames(self, gals_fname, voids_fname) :
        # assumes gals_fname is binary file f8 ra, dec, z
        # and voids_fname is the sky positions file from VIDE
        return self.compute_from_arrays(*np.fromfile(gals_fname).reshape(3, -1),
                                        *np.loadtxt(voids_fname, usecols=(0,1,2,3,), unpack=True))



    def nz(self, z) :
        _, edges = scott_bin_width(z, return_bins=True)
        dig = np.searchsorted(edges, z, 'right')
        N = np.bincount(dig, minlength=len(edges)+1)[1:-1]
        R_hi = self.cosmo.comoving_distance(edges[1:])
        R_lo = self.cosmo.comoving_distance(edges[:-1])
        dV = 4*np.pi/3 * (R_hi**3 - R_lo**3) * PLKCalc.fsky
        return InterpolatedUnivariateSpline(0.5*(edges[1:]+edges[:-1]), N/dV, ext='const')


    def interval(self, N) :
        per_rank = N // self.comm.size
        lo = self.comm.rank * per_rank
        hi = (self.comm.rank+1)*per_rank if self.comm.rank != (self.comm.size-1) else N
        return lo, hi


    def rand_voids(self, z_voids) :
        # construct a probability distribution for the randoms redshift distribution
        kernel_density = KernelDensity(bandwidth=scott_bin_width(z_voids)).fit(z_voids.reshape(-1, 1))

        N_rand_voids = PLKCalc.N_rand_voids // self.comm.size
        z_rand_voids = kernel_density.sample(N_rand_voids, random_state=self.rng)

        # here we are assuming that the cut according to minimum radius has already been
        # performed
        ra_dec_rand_voids = self.rng.choice(self.ra_dec_collected_voids, size=N_rand_voids)
        return *ra_dec_rand_voids.T, z_rand_voids
