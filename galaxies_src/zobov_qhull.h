#ifndef ZOBOV_QHULL_H
#define ZOBOV_QHULL_H

#include <cstdint>

#include "zobov.h"

#include "libqhullcpp/Qhull.h"

int delaunadj (coordT *x, int64_t nvp, int64_t nvpbuf, int64_t nvpall, PartAdj **adjs);
int vorvol (coordT *deladjs, coordT *points, pointT *intpoints, int64_t numpoints, float *vol);

#endif // ZOBOV_QHULL_H
