#ifndef READ_FOF_H
#define READ_FOF_H

#include <cstdint>

#include "enums.h"

#include "read_fof.h"

template<bool have_vel>
int read_fof (const char *dirname,
              float *BoxSize, float *z, int64_t *N,
              float **M, float **pos,
              [[maybe_unused]] float **vel,
              [[maybe_unused]] float **sec);


#endif
