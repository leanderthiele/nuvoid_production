#ifndef POPULATE_H
#define POPULATE_H

#include <cstdint>
#include <vector>

#include "have_if.h"
#include "halo.h"
#include "hod_params.h"
#include "globals.h"

template<bool have_abias, VelMode vmode, bool have_zdep>
int populate (Globals &globals, const HODParams<have_abias, vmode==VelMode::Biased, have_zdep> &hod_params,
              std::vector<Halo<have_abias>> &halos,
              std::vector<float> &xgal,
              HAVE_IF(vmode != VelMode::None, std::vector<float> &) vgal,
              HAVE_IF(vmode != VelMode::None, std::vector<float> &) vhlo);

#endif // POPULATE_H
