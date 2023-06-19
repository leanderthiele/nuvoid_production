#ifndef ZOBOV_H
#define ZOBOV_H

#include <cstdint>

struct ZobovCfg
{
    // free parameters
    int numdiv;
    double border;

    // computed
    double width, width2, totwidth, totwidth2, bf, s, g;
    int64_t nvpmin, nvpmax, nvpbufmin, nvpbufmax;

    ZobovCfg (int numdiv_, double border_)
        : numdiv {numdiv_}, border {border_}
    { }
};

struct PartAdj { int nadj, *adj; };

#endif // ZOBOV_H
