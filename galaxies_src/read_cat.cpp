#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>

#include "enums.h"
#include "globals.h"
#include "halo.h"
#include "read_hdf5.h"
#include "read_bigfile.h"
#include "read_fof.h"
#include "err.h"
#include "timing.h"

#include "read_cat.h"

template<typename T=float>
int find_cosmo_par (std::FILE *f, const char *name, T *out, const char *fmt="%g")
{
    std::fseek(f, 0, SEEK_SET);
    char buffer[64];
    std::sprintf(buffer, "%s=%s", name, fmt);

    char line[256];

    bool found = false;
    while (std::fgets(line, sizeof(line), f))
        if (std::sscanf(line, buffer, out) == 1) { found=true; break; }

    CHECK(!found, return 1);
    return 0;
}

int parse_cosmo_info (const char *base, Globals &globals)
{
    char buffer[512];
    sprintf(buffer, "%s/cosmo.info", base);
    auto f = std::fopen(buffer, "r");

    int status = 0;
    status |= find_cosmo_par(f, "Omega_m", &globals.O_m);
    status |= find_cosmo_par(f, "Omega_b", &globals.O_b);
    status |= find_cosmo_par(f, "Omega_cdm", &globals.O_cdm);
    status |= find_cosmo_par(f, "Omega_nu", &globals.O_nu);
    status |= find_cosmo_par(f, "h", &globals.h);
    status |= find_cosmo_par(f, "n_s", &globals.n_s);
    status |= find_cosmo_par(f, "A_s", &globals.A_s);
    status |= find_cosmo_par(f, "sigma_8", &globals.sigma_8);

    char hash_str[256];
    status |= find_cosmo_par(f, "hash", hash_str, "%s");
    int found;
    // low word
    found = std::sscanf(hash_str+(std::strlen(hash_str)-16), "%lx", &globals.cosmo_hash_low_word);
    status |= (found != 1);
    // high word
    hash_str[std::strlen(hash_str)-16] = 0;
    found = std::sscanf(hash_str, "%lx", &globals.cosmo_hash_high_word);
    status |= (found != 1);

    std::fclose(f);

    CHECK(status, return 1);

    return 0;
}

template<Cat cat_type, Sec secondary, bool have_vel>
int read_cat (Globals &globals,
              std::vector<Halo<secondary != Sec::None>> &halos)
{
    static_assert(!((cat_type==Cat::FOF || cat_type==Cat::RFOF || cat_type==Cat::OLDFOF)
                  && secondary != Sec::None));

    int status;
    
    auto start = start_time();

    status = parse_cosmo_info(globals.base, globals);
    CHECK(status, return 1);

    float time = 1.0/(1.0+globals.z);
    char buffer[512];

    if constexpr (cat_type == Cat::OLDFOF)
        // FIXME this is very hacky...
        sprintf(buffer, "%s/groups_003", globals.base);
    else
        sprintf(buffer, "%s/%s_%.4f", globals.base,
                (cat_type==Cat::Rockstar) ? "rockstar"
                : (cat_type==Cat::FOF) ? "fof"
                : (cat_type==Cat::RFOF) ? "rfof"
                : "none",
                time);

    float other_z;
    float *M=nullptr, *pos=nullptr, *vel=nullptr, *sec=nullptr;

    auto read_f = (cat_type == Cat::OLDFOF) ? read_fof<have_vel>
                                            : read_bigfile<cat_type, secondary, have_vel>;
    
    status = read_f(buffer, &globals.BoxSize, &other_z, &globals.Nhalos, &M, &pos,
                    (have_vel) ? &vel : nullptr, (secondary != Sec::None) ? &sec : nullptr);
    
    CHECK(status, return 1);
    CHECK(std::fabs(globals.z-other_z) > 1e-2, return 1);

    halos.reserve(globals.Nhalos);

    for (int64_t ii=0; ii<globals.Nhalos; ++ii)
    {
        Halo<secondary != Sec::None> tmp;
        tmp.M = M[ii];
        for (int64_t jj=0; jj<3; ++jj)
        {
            tmp.pos[jj] = pos[3*ii+jj];

            if constexpr (have_vel)
                tmp.vel[jj] = vel[3*ii+jj];
        }
        if constexpr (secondary != Sec::None) tmp.abias_property = sec[ii];
        halos.push_back(tmp);
    }

    if (M) std::free(M);
    if (pos) std::free(pos);
    if (vel) std::free(vel);
    if (sec) std::free(sec);

    globals.time_read_cat = get_time(start);

    return 0;
}

// instantiate
#define INSTANTIATE(cat_type, secondary, have_vel) \
    template int read_cat<cat_type, secondary, have_vel> \
    (Globals &, std::vector<Halo<secondary != Sec::None>> &)

INSTANTIATE(Cat::OLDFOF, Sec::None, true);
INSTANTIATE(Cat::OLDFOF, Sec::None, false);
INSTANTIATE(Cat::FOF, Sec::None, true);
INSTANTIATE(Cat::FOF, Sec::None, false);
INSTANTIATE(Cat::RFOF, Sec::None, true);
INSTANTIATE(Cat::RFOF, Sec::None, false);
INSTANTIATE(Cat::Rockstar, Sec::None, true);
INSTANTIATE(Cat::Rockstar, Sec::None, false);
INSTANTIATE(Cat::Rockstar, Sec::Conc, true);
INSTANTIATE(Cat::Rockstar, Sec::Conc, false);
INSTANTIATE(Cat::Rockstar, Sec::TU, true);
INSTANTIATE(Cat::Rockstar, Sec::TU, false);

#undef INSTANTIATE
