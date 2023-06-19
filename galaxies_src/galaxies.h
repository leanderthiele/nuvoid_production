#ifndef GALAXIES_H
#define GALAXIES_H

#include <vector>
#include <cstdint>

#include "have_if.h"
#include "enums.h"
#include "hod_params.h"
#include "globals.h"

template<Cat cat_type, Sec secondary, VelMode vmode, bool have_zdep>
int get_galaxies (Globals &globals,
                  const HODParams<secondary != Sec::None, vmode==VelMode::Biased, have_zdep> &hod_params,
                  std::vector<float> &xgal,
                  HAVE_IF(vmode != VelMode::None, std::vector<float> &) vgal,
                  HAVE_IF(vmode != VelMode::None, std::vector<float> &) vhlo);

#endif
