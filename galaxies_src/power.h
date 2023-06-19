#ifndef POWER_H
#define POWER_H

#include <cstdint>
#include <vector>

#include "have_if.h"
#include "enums.h"
#include "globals.h"

template<MAS mas, RSD rsd, bool have_mark>
int get_power (Globals &globals,
               const std::vector<float> &xgal,
               HAVE_IF(rsd != RSD::None, const std::vector<float> &) vgal,
               HAVE_IF(have_mark, float) p,
               HAVE_IF(have_mark, float) delta_s,
               HAVE_IF(have_mark, float) R,
               int64_t Nmesh, float kmax,
               std::vector<float> &k, std::vector<float> &Pk, std::vector<int64_t> &Nk);

#endif
