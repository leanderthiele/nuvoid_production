#ifndef READ_BIGFILE_H
#define READ_BIGFILE_H

#include <cstdint>

#include "enums.h"

template<Cat cat_type, Sec secondary, bool have_vel>
int read_bigfile (const char *dirname,
                  float *BoxSize, float *z, int64_t *N,
                  float **M, float **pos,
                  [[maybe_unused]] float **vel,
                  [[maybe_unused]] float **sec);


#endif
