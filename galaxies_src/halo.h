#ifndef HALO_H
#define HALO_H

#include <cstdint>

#include "have_if.h"

template<bool have_abias>
struct Halo
{
    float M;
    float pos[3];
    float vel[3];

    HAVE_IF(have_abias, float) abias_property;
    HAVE_IF(have_abias, bool) type; // type 1 or 2
};

#endif // HALO_H
