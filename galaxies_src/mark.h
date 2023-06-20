#ifndef MARK_H
#define MARK_H

#include <cstdint>
#include <vector>

#include "globals.h"

int mark (Globals &globals, std::vector<float> &m,
          float p, float delta_s, float R);

#endif // MARK_H
