#include <vector>
#include <cstdio>

#include "err.h"
#include "enums.h"
#include "globals.h"
#include "read_cat.h"


int main (int argc, char **argv)
{
    int status = 0;

    Globals globals ("/home/lthiele/cd3globus/Quijote_BSQ/fof/1", /*z=*/0.0);

    std::vector<Halo</*have_abias=*/false>> halos;

    status = read_cat<Cat::OLDFOF, Sec::None, /*have_vel=*/true>(globals, halos);
    CHECK(status, return 1);

    return 0;
}
