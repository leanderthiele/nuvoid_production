#ifndef PK_H
#define PK_H

#include <cstdint>
#include <vector>

#include "enums.h"
#include "globals.h"

template<MAS mas, RSD rsd>
int pk (Globals &globals, const std::vector<float> &m,
        float kmax,
        std::vector<float> &k, std::vector<float> &Pk, std::vector<int64_t> &Nk);

#endif
