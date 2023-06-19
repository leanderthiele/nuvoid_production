#ifndef READ_CAT_H
#define READ_CAT_H

#include <vector>

#include "enums.h"
#include "globals.h"
#include "halo.h"

template<Cat cat_type, Sec secondary, bool have_vel>
int read_cat (Globals &globals,
              std::vector<Halo<secondary != Sec::None>> &halos);

#endif // READ_CAT_H
