/* Contains global properties of the simulation */

#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>
#include <cstdint>
#include <limits>

#include "enums.h"

struct Globals
{
    const char *base;
    float z;
    HaloDef mdef;
    int64_t seed;
    float BoxSize=-1;
    // the hash is 128 bit which is a bit annoying...
    uint64_t cosmo_hash_low_word, cosmo_hash_high_word;
    float O_m=-1, O_b=-1, O_nu=-1, O_cdm=-1, h=-1, n_s=-1, sigma_8=-1, A_s=-1;
    int64_t Nhalos=-1, Ngals=-1, Ncen=-1, Nsat=-1;

    float time_get_galaxies=-1,
            time_read_cat=-1,
            time_populate=-1,
              time_assign_types=-1,
              time_draw_gals=-1,
          time_mark=-1,
          time_power=-1,
            time_mesh=-1,
            time_pk=-1,
              time_fft=-1,
              time_bin=-1;
          

    Globals (const char *base_, float z_,
             int64_t seed_=std::numeric_limits<int64_t>::max(),
             HaloDef mdef_=HaloDef::v)
        : base {base_}, z {z_}, mdef {mdef_}, seed {seed_}
    {
        if (seed == std::numeric_limits<int64_t>::max())
            seed = std::random_device()();
    }

    Globals () {}
};

#endif // GLOBALS_H
