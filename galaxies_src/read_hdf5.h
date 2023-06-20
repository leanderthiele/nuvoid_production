#ifndef READ_HDF5_H
#define READ_HDF5_H

#include <cstdint>

#include "enums.h"

template<Sec secondary, bool have_vel>
int read_hdf5 (const char *dirname, int have_subs,
               float *BoxSize, float *z, int64_t *N,
               int64_t **id, float **M, float **pos,
               [[maybe_unused]] float **vel,
               [[maybe_unused]] float **sec);


#endif
