#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <glob.h>

#include "have_if.h"
#include "enums.h"
#include "err.h"



// FIXME this is very hacky and only for the very specific application
template<bool have_vel>
int read_fof (const char *dirname,
              float *BoxSize, float *z, int64_t *N,
              float **M, float **pos,
              [[maybe_unused]] float **vel,
              [[maybe_unused]] float **sec)
{
    int status;

    // FIXME this is where we are assuming a lot
    *BoxSize = 1e3;
    *z = 0.0;

    // initialize
    *N = 0;

    // find the files
    char buffer[512];
    glob_t glob_result;

    sprintf(buffer, "%s/group_tab_*[0-9].*[0-9]", dirname);
    status = glob(buffer, GLOB_TILDE_CHECK, NULL, &glob_result);

    CHECK(status, return 1);
    CHECK(glob_result.gl_pathc == 0, return 1);

    float *M_, *pos_, *vel_;

    for (int ii=0; ii<glob_result.gl_pathc; ++ii)
    {
        int32_t Ngroups, TotNgroups, Nids, Nfiles;
        uint64_t TotNids;

        auto fp = std::fopen(glob_result.gl_pathv[ii], "r");
        
        std::fread(&Ngroups, sizeof Ngroups, 1, fp);
        std::fread(&TotNgroups, sizeof TotNgroups, 1, fp);
        std::fread(&Nids, sizeof Nids, 1, fp);
        std::fread(&TotNids, sizeof TotNids, 1, fp);
        std::fread(&Nfiles, sizeof Nfiles, 1, fp);

        CHECK(Nfiles != glob_result.gl_pathc, return 1);

        if (!ii)
        {
            *M = (float *)std::malloc(TotNgroups * sizeof(float));
            *pos = (float *)std::malloc(3 * TotNgroups * sizeof(float));
            if constexpr (have_vel)
                *vel = (float *)std::malloc(3 * TotNgroups * sizeof(float));
        }

        M_ = *M + *N;
        pos_ = *pos + *N;
        if constexpr (have_vel)
            vel_ = *vel + *N;

        // order important here!
        *N += Ngroups;

        if (ii == glob_result.gl_pathc-1)
            CHECK(*N != TotNgroups, return 1);

        std::fseek(fp, Ngroups * sizeof(int32_t), SEEK_CUR); // skip GroupLen
        std::fseek(fp, Ngroups * sizeof(int32_t), SEEK_CUR); // skip GroupOffset
        std::fread(M_, sizeof(float), Ngroups, fp);
        std::fread(pos_, sizeof(float), 3*Ngroups, fp);
        if constexpr (have_vel)
            std::fread(vel_, sizeof(float), 3*Ngroups, fp);

        std::fclose(fp);
    }

    globfree(&glob_result);

    for (int ii=0; ii<*N; ++ii)
    {
        (*M)[ii] *= 1e10; // convert to Msun/h
        for (int jj=0; jj<3; ++jj)
        {
            (*pos)[3*ii+jj] *= 1e-3; // convert to Mpc/h
            if constexpr (have_vel)
                (*vel)[3*ii+jj] *= (1.0 + *z); // convert to physical km/s (according to Paco's code)
        }
    }

    return 0;
}


#define INSTANTIATE(have_vel) \
    template int read_fof<have_vel> \
    (const char *, float *, float *, int64_t *, \
     float **, float **, float **, float **)

INSTANTIATE(true);
INSTANTIATE(false);
