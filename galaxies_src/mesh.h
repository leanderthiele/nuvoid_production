#ifndef MESH_H
#define MESH_H

#include <cstdint>
#include <vector>

#include "enums.h"
#include "globals.h"

// adds to allocated mesh
template<MAS mas, RSD rsd>
int mesh (Globals &globals, const float *rp, const float *vp/*pass nullptr if no RSD*/,
          int64_t Nm, std::vector<float> &m);

#endif
