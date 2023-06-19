/* Main driver code */

#include <cstdio>
#include <cstdint>
#include <vector>

#include "enums.h"
#include "have_if.h"
#include "globals.h"
#include "hod_params.h"
#include "read_cat.h"
#include "populate.h"
#include "err.h"
#include "timing.h"

#include "galaxies.h"

template<Cat cat_type, Sec secondary, VelMode vmode, bool have_zdep>
int get_galaxies (Globals &globals,
                  const HODParams<secondary != Sec::None, vmode==VelMode::Biased, have_zdep> &hod_params,
                  std::vector<float> &xgal,
                  HAVE_IF(vmode != VelMode::None, std::vector<float> &) vgal,
                  HAVE_IF(vmode != VelMode::None, std::vector<float> &) vhlo)
{
    int status;

    auto start = start_time();

    std::vector<Halo<secondary != Sec::None>> halos;
    status = read_cat<cat_type, secondary, vmode != VelMode::None>(globals, halos);
    CHECK(status, return 1);

    status = populate<secondary != Sec::None, vmode, have_zdep>
        (globals, hod_params, halos, xgal, vgal, vhlo);
    CHECK(status, return 1);

    globals.time_get_galaxies = get_time(start);

    return 0;
}

// instantiate
#define INSTANTIATE(cat_type, secondary, vmode, have_zdep) \
    template int get_galaxies<cat_type, secondary, vmode> \
    (Globals &, \
     const HODParams<secondary != Sec::None, vmode==VelMode::Biased, have_zdep> &, \
     std::vector<float> &, \
     HAVE_IF(vmode != VelMode::None, std::vector<float> &), \
     HAVE_IF(vmode != VelMode::None, std::vector<float> &))

INSTANTIATE(Cat::FOF, Sec::None, VelMode::None, false);
INSTANTIATE(Cat::FOF, Sec::None, VelMode::Biased, false);
INSTANTIATE(Cat::FOF, Sec::None, VelMode::Unbiased, false);
INSTANTIATE(Cat::RFOF, Sec::None, VelMode::None, false);
INSTANTIATE(Cat::RFOF, Sec::None, VelMode::Biased, false);
INSTANTIATE(Cat::RFOF, Sec::None, VelMode::Unbiased, false);
INSTANTIATE(Cat::Rockstar, Sec::None, VelMode::None, false);
INSTANTIATE(Cat::Rockstar, Sec::None, VelMode::Biased, false);
INSTANTIATE(Cat::Rockstar, Sec::None, VelMode::Unbiased, false);
INSTANTIATE(Cat::Rockstar, Sec::Conc, VelMode::None, false);
INSTANTIATE(Cat::Rockstar, Sec::Conc, VelMode::Biased, false);
INSTANTIATE(Cat::Rockstar, Sec::Conc, VelMode::Unbiased, false);
INSTANTIATE(Cat::Rockstar, Sec::TU, VelMode::None, false);
INSTANTIATE(Cat::Rockstar, Sec::TU, VelMode::Biased, false);
INSTANTIATE(Cat::Rockstar, Sec::TU, VelMode::Unbiased, false);
INSTANTIATE(Cat::FOF, Sec::None, VelMode::None, true);
INSTANTIATE(Cat::FOF, Sec::None, VelMode::Biased, true);
INSTANTIATE(Cat::FOF, Sec::None, VelMode::Unbiased, true);
INSTANTIATE(Cat::RFOF, Sec::None, VelMode::None, true);
INSTANTIATE(Cat::RFOF, Sec::None, VelMode::Biased, true);
INSTANTIATE(Cat::RFOF, Sec::None, VelMode::Unbiased, true);
INSTANTIATE(Cat::Rockstar, Sec::None, VelMode::None, true);
INSTANTIATE(Cat::Rockstar, Sec::None, VelMode::Biased, true);
INSTANTIATE(Cat::Rockstar, Sec::None, VelMode::Unbiased, true);
INSTANTIATE(Cat::Rockstar, Sec::Conc, VelMode::None, true);
INSTANTIATE(Cat::Rockstar, Sec::Conc, VelMode::Biased, true);
INSTANTIATE(Cat::Rockstar, Sec::Conc, VelMode::Unbiased, true);
INSTANTIATE(Cat::Rockstar, Sec::TU, VelMode::None, true);
INSTANTIATE(Cat::Rockstar, Sec::TU, VelMode::Biased, true);
INSTANTIATE(Cat::Rockstar, Sec::TU, VelMode::Unbiased, true);
