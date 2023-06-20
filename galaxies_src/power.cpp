#include <cstdint>
#include <cstdio>
#include <vector>

#include "have_if.h"
#include "enums.h"
#include "globals.h"
#include "mesh.h"
#include "mark.h"
#include "pk.h"
#include "err.h"
#include "timing.h"

#include "power.h"

template<MAS mas, RSD rsd, bool have_mark>
int get_power (Globals &globals,
               const std::vector<float> &xgal,
               HAVE_IF(rsd != RSD::None, const std::vector<float> &) vgal,
               HAVE_IF(have_mark, float) p,
               HAVE_IF(have_mark, float) delta_s,
               HAVE_IF(have_mark, float) R,
               int64_t Nmesh, float kmax,
               std::vector<float> &k, std::vector<float> &Pk, std::vector<int64_t> &Nk)
{
    int status;

    auto start = start_time();

    const float *v;
    if constexpr (rsd==RSD::None) v = nullptr;
    else v = vgal.data();

    std::vector<float> m; // holds the mesh
    status = mesh<mas, rsd>(globals, xgal.data(), v, Nmesh, m);
    CHECK(status, return 1);

    if constexpr (have_mark)
    {
        status = mark(globals, m, p, delta_s, R); 
        CHECK(status, return 1);
    }

    status = pk<mas, rsd>(globals, m, kmax, k, Pk, Nk);
    CHECK(status, return 1);

    globals.time_power = get_time(start);

    return 0;
}

#define INSTANTIATE(mas, rsd, have_mark) \
    template int get_power<mas, rsd, have_mark> \
    (Globals &, \
     const std::vector<float> &, \
     HAVE_IF(rsd != RSD::None, const std::vector<float> &), \
     HAVE_IF(have_mark, float), HAVE_IF(have_mark, float), HAVE_IF(have_mark, float), \
     int64_t, float, \
     std::vector<float> &, std::vector<float> &, std::vector<int64_t> &)

INSTANTIATE(MAS::NGP, RSD::None, false);
INSTANTIATE(MAS::NGP, RSD::x, false);
INSTANTIATE(MAS::NGP, RSD::y, false);
INSTANTIATE(MAS::NGP, RSD::z, false);
INSTANTIATE(MAS::CIC, RSD::None, false);
INSTANTIATE(MAS::CIC, RSD::x, false);
INSTANTIATE(MAS::CIC, RSD::y, false);
INSTANTIATE(MAS::CIC, RSD::z, false);
INSTANTIATE(MAS::TSC, RSD::None, false);
INSTANTIATE(MAS::TSC, RSD::x, false);
INSTANTIATE(MAS::TSC, RSD::y, false);
INSTANTIATE(MAS::TSC, RSD::z, false);
INSTANTIATE(MAS::PCS, RSD::None, false);
INSTANTIATE(MAS::PCS, RSD::x, false);
INSTANTIATE(MAS::PCS, RSD::y, false);
INSTANTIATE(MAS::PCS, RSD::z, false);
INSTANTIATE(MAS::NGP, RSD::None, true);
INSTANTIATE(MAS::NGP, RSD::x, true);
INSTANTIATE(MAS::NGP, RSD::y, true);
INSTANTIATE(MAS::NGP, RSD::z, true);
INSTANTIATE(MAS::CIC, RSD::None, true);
INSTANTIATE(MAS::CIC, RSD::x, true);
INSTANTIATE(MAS::CIC, RSD::y, true);
INSTANTIATE(MAS::CIC, RSD::z, true);
INSTANTIATE(MAS::TSC, RSD::None, true);
INSTANTIATE(MAS::TSC, RSD::x, true);
INSTANTIATE(MAS::TSC, RSD::y, true);
INSTANTIATE(MAS::TSC, RSD::z, true);
INSTANTIATE(MAS::PCS, RSD::None, true);
INSTANTIATE(MAS::PCS, RSD::x, true);
INSTANTIATE(MAS::PCS, RSD::y, true);
INSTANTIATE(MAS::PCS, RSD::z, true);
