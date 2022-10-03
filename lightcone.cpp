#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>

#include "cuboid.h"

// these are the possible remaps I found
const int remaps[][9] =
                  { // 1.4142 1.0000 0.7071
                    { 1, 1, 0,
                      0, 0, 1,
                      1, 0, 0, },
                    // 1.7321 0.8165 0.7071
                    { 1, 1, 1,
                      1, 0, 0,
                      0, 1, 0, },
                  };

// get from the quadrant ra=[-90,90], dec=[0,90] to the NGC footprint
// we only need a rotation around the y-axis I believe
const double alpha = 124.0 * M_PI / 180.0;

// in units of L1, L2, L3
const double origin[] = { 0.5, -0.058, 0.0 };

template<bool reverse>
int dbl_cmp (const void *a_, const void *b_)
{
    double a = *(double *)a_;
    double b = *(double *)b_;
    int sgn = (reverse) ? -1 : 1;
    return sgn * ( (a>b) ? +1 : -1 );
}

double integrand (double z, void *p)
{
    static const double H0inv = 2.99792458e3; // Mpc/h
    double Omega_m = *(double *)p;
    return H0inv / std::sqrt( Omega_m*gsl_pow_3(1.0+z)+(1.0-Omega_m) );
}

void comoving (double Omega_m, int N, double *z, double *chi)
// populate chi with chi(z) in Mpc/h
{
    gsl_function F;
    F.function = integrand;
    F.params = &Omega_m;

    static const size_t ws_size = 1024;
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(ws_size);

    double err;
    for (int ii=0; ii<N; ++ii)
        gsl_integration_qag(&F, 0.0, z[ii], 1e-2, 0.0, ws_size, 6, ws, chi+ii, &err);

    gsl_integration_workspace_free(ws);
}

template<typename T>
inline double per_unit (T x, double BoxSize)
{
    double x1 = (double)x / BoxSize;
    x1 = std::fmod(x, 1.0);
    return (x1<0.0) ? x1+1.0 : x1;
}

int main (int argc, char **argv)
{
    char **c = argv + 1;

    char *inpath = *(c++);
    char *inident = *(c++);
    char *outident = *(c++);
    double BoxSize = std::atof(*(c++));
    double Omega_m = std::atof(*(c++));
    double zmin = std::atof(*(c++));
    double zmax = std::atof(*(c++));
    int remap_case = std::atoi(*(c++));

    int Nsnaps = 0;
    double times[64];
    while (*c)
        times[Nsnaps++] = std::atof(*(c++));

    // get increasing redshifts
    std::qsort(times, Nsnaps, sizeof(double), dbl_cmp</*reverse=*/true>);

    double redshifts[Nsnaps];
    for (int ii=0; ii<Nsnaps; ++ii)
        redshifts[ii] = 1.0/times[ii] - 1.0;

    // define the redshift boundaries
    // TODO we can maybe do something more sophisticated here
    double redshift_bounds[Nsnaps+1];
    redshift_bounds[0] = zmin;
    redshift_bounds[Nsnaps] = zmax;
    for (int ii=1; ii<Nsnaps; ++ii)
        redshift_bounds[ii] = 0.5*(redshifts[ii-1]+redshifts[ii]);
    
    // convert to comoving distances
    double chi_bounds[Nsnaps+1];
    comoving(Omega_m, Nsnaps+1, redshift_bounds, chi_bounds);

    // initialize the interpolator getting us from comoving to redshift
    const int Ninterp = 1024;
    double z_interp_min = zmin-0.01, z_interp_max=zmax+0.01;
    double z_interp[Ninterp];
    double chi_interp[Ninterp];
    for (int ii=0; ii<Ninterp; ++ii)
        z_interp[ii] = z_interp_min + (z_interp_max-z_interp_min)*(double)ii/(double)(Ninterp-1);
    comoving(Omega_m, Ninterp, z_interp, chi_interp);
    gsl_interp *z_chi_interp = gsl_interp_alloc(gsl_interp_cspline, Ninterp);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_interp_init(z_chi_interp, chi_interp, z_interp, Ninterp);

    // initialize the transformation
    auto C = Cuboid(remaps[remap_case]);

    // this is the output
    std::vector<double> ra_out, dec_out, z_out;

    for (int ii=0; ii<Nsnaps; ++ii)
    {
        char fname[512];
        std::sprintf(fname, "%s/galaxies/galaxies_%s_%.4f.bin", inpath, inident, times[ii]);

        auto fp = std::fopen(fname, "rb");

        // figure out number of galaxies
        std::fseek(fp, 0, SEEK_END);
        auto nbytes = std::ftell(fp);

        auto Ngal = (nbytes/sizeof(float)-1/*rsd factor*/)/6;
        if (!((6*Ngal+1)*sizeof(float)==nbytes)) return 1;

        // go back to beginning
        std::fseek(fp, 0, SEEK_SET);

        float rsd_factor_f;
        std::fread(&rsd_factor_f, sizeof(float), 1, fp);

        float *xgal_f = (float *)std::malloc(3 * Ngal * sizeof(float));
        float *vgal_f = (float *)std::malloc(3 * Ngal * sizeof(float));
        std::fread(xgal_f, sizeof(float), 3*Ngal, fp);
        std::fread(vgal_f, sizeof(float), 3*Ngal, fp);

        // these will contain the outputs
        double *xgal = (double *)std::malloc(3 * Ngal * sizeof(double));
        double *vgal = (double *)std::malloc(3 * Ngal * sizeof(double));

        // now transform the positions and velocities
        for (size_t jj=0; jj<Ngal; ++jj)
        {
            C.Transform(per_unit(xgal_f[3*jj+0], BoxSize),
                        per_unit(xgal_f[3*jj+1], BoxSize),
                        per_unit(xgal_f[3*jj+2], BoxSize),
                        xgal[3*jj+0], xgal[3*jj+1], xgal[3*jj+2]);
            for (int kk=0; kk<3; ++kk) xgal[3*jj+kk] *= BoxSize;

            C.VelocityTransform(vgal_f[3*jj+0], vgal_f[3*jj+1], vgal_f[3*jj+2],
                                vgal[3*jj+0], vgal[3*jj+1], vgal[3*jj+2]);
        }

        // now perform RSD
        double L[] = { C.L1, C.L2, C.L3 };
        for (size_t jj=0; jj<Ngal; ++jj)
        {
            // compute the line-of-sight vector
            double los[3];
            for (int kk=0; kk<3; ++kk) los[kk] = xgal[3*jj+kk] - origin[kk]*BoxSize*L[kk];

            // compute length of the line-of-sight vector
            double abs_los = std::hypot(los[0], los[1], los[2]);

            // compute the velocity projection onto the line of sight
            double vproj = (los[0]*vgal[3*jj+0]+los[1]*vgal[3*jj+1]+los[2]*vgal[3*jj+2])
                           / abs_los;

            for (int kk=0; kk<3; ++kk) xgal[3*jj+kk] += rsd_factor_f * vproj * los[kk] / abs_los;
        }
        
        // choose the galaxies within this redshift shell
        for (size_t jj=0; jj<Ngal; ++jj)
        {
            double los[3];
            for (int kk=0; kk<3; ++kk) los[kk] = xgal[3*jj+kk] - origin[kk]*BoxSize*L[kk];
            
            double chi = std::hypot(los[0], los[1], los[2]);

            if (chi>chi_bounds[ii] && chi<chi_bounds[ii+1])
            // we are in the comoving shell that's coming from this snapshot
            {
                // rotate the line of sight into the NGC footprint and transpose the axes into
                // canonical order
                double x1, x2, x3;
                x1 = std::cos(alpha) * los[2] - std::sin(alpha) * los[1];
                x2 = los[0];
                x3 = std::sin(alpha) * los[2] + std::cos(alpha) * los[1];

                double z = gsl_interp_eval(z_chi_interp, chi_interp, z_interp, chi, acc);
                double theta = std::acos(x3/chi);
                double phi = std::atan2(x2, x1);

                z_out.push_back(z);
                dec_out.push_back(90.0-theta/M_PI*180.0);
                ra_out.push_back(phi/M_PI*180.0);
            }
        }
    }

    // output
    char fname[512];
    std::sprintf(fname, "%s/galaxies/lightcone_%s_%s.txt", inpath, inident, outident);
    auto fp = std::fopen(fname, "w");
    std::fprintf(fp, "# RA, DEC, z\n");
    for (size_t ii=0; ii<z_out.size(); ++ii)
        std::fprintf(fp, "%.8f %.8f %.8f\n", ra_out[ii], dec_out[ii], z_out[ii]);

    // clean up
    gsl_interp_free(z_chi_interp);
    gsl_interp_accel_free(acc);

    return 0;
}
