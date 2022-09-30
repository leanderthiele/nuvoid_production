#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>

#include "cuboid.h"

// these are the possible remaps I found
int remaps[][9] = { { 1, 1, 0,
                      0, 0, 1,
                      1, 0, 0, },
                    { 1, 1, 1,
                      1, 0, 0,
                      0, 1, 0, },
                  };

template<bool reverse>
int dbl_cmp (const void *a_, const void *b_)
{
    double a = *(double *)a_;
    double b = *(double *)b_;
    int sgn = (reverse) ? -1 : 1;
    return sgn * ( (a>b) ? +1 : -1 );
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
    char *inpattern = *(c++);
    double BoxSize = std::atof(*(c++));
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

    // initialize the transformation
    auto C = Cuboid(remaps[remap_case]);

    // this is the output
    std::vector<double> ra, dec, z;

    for (int ii=0; ii<Nsnaps; ++ii)
    {
        char basename[512], fname[512];
        std::sprintf(basename, inpattern, times[ii]);
        std::sprintf(fname, "%s/%s", inpath, inpattern);

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

        // implement RSD
        for (size_t jj=0; jj<Ngal; ++jj)
        {
        // TODO can't be arsed right now
        }
    }

    return 0;
}
