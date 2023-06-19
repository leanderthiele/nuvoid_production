#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "enums.h"
#include "globals.h"
#include "timing.h"

#include "mesh.h"

#define SQUARE(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

template<MAS mas>
constexpr int64_t Nkernel = -1;

template<> constexpr int64_t Nkernel<MAS::NGP> = 1;
template<> constexpr int64_t Nkernel<MAS::CIC> = 2;
template<> constexpr int64_t Nkernel<MAS::TSC> = 3;
template<> constexpr int64_t Nkernel<MAS::PCS> = 4;

static inline int64_t periodic (int64_t i, int64_t N)
{
    int64_t r = i % N;
    return (r<0) ? r+N : r;
}

static inline int64_t to_1d (int64_t ix, int64_t iy, int64_t iz, int64_t N)
{
    ix = periodic(ix, N);
    iy = periodic(iy, N);
    iz = periodic(iz, N);
    return ix*N*N + iy*N + iz;
}

template<MAS mas>
static inline int64_t kernel (float x, float *k);
// writes the kernel evaluation into the output array (of length Nkernel),
// and returns the first index (in this specific direction)

template<> int64_t kernel<MAS::NGP> (float x, float *k)
{
    k[0] = 1.0F;
    return std::lround(x);
}

template<> int64_t kernel<MAS::CIC> (float x, float *k)
{
    float sl = x - std::floor(x);
    k[0] = 1.0F - sl;
    k[1] = sl;
    return (int64_t)std::floor(x);
}

template<> int64_t kernel<MAS::TSC> (float x, float *k)
{
    float s = x - std::round(x);
    k[1] = 0.75F - SQUARE(s);
    k[0] = 0.5F * SQUARE( 0.5F - s );
    k[2] = 0.5F * SQUARE( 0.5F + s );
    return std::lround(x) - 1;
}

template<> int64_t kernel<MAS::PCS> (float x, float *k)
{
    float sl = x - std::floor(x);
    #define SIXTH 0.16666666666667F
    k[1] = SIXTH * ( 4.0F - 6.0F * SQUARE(sl) + 3.0F * CUBE(sl) );
    k[2] = SIXTH * ( 4.0F - 6.0F * SQUARE(1.0F-sl) + 3.0F * CUBE(1.0F-sl) );
    k[0] = SIXTH * CUBE( 1.0F - sl );
    k[3] = SIXTH * CUBE( sl );
    #undef SIXTH
    return (int64_t)std::floor(x) - 1;
}

template<MAS mas, RSD rsd>
int mesh (Globals &globals, const float *rp, const float *vp, int64_t Nm, std::vector<float> &m)
// peforms the mesh allocation
// the input rp do not need to have periodic boundary conditions imposed
// (but they need to be in a reasonable range I believe)
{
    auto start = start_time();

    m.resize(Nm * Nm * Nm, 0.0F);

    [[maybe_unused]] float rsd_factor;
    if constexpr (rsd != RSD::None)
        rsd_factor = (1.0F+globals.z)
                     / (100.0F * std::sqrt(globals.O_m*CUBE(1.0F+globals.z) + (1.0F-globals.O_m)));

    for (int64_t ii=0; ii<globals.Ngals; ii++)
    {
        const float *r = rp + 3 * ii;

        // note not dereferencing is ok with nullptr
        [[maybe_unused]] const float *v = vp + 3 * ii;

        float k[3][Nkernel<mas>];
        int64_t idx_min[3];
        for (int jj=0; jj<3; ++jj)
        {
            float rj = r[jj];
            
            if constexpr (rsd != RSD::None)
                if (jj==(int)rsd)
                    rj += rsd_factor * v[jj];

            float rnorm = rj * (Nm/globals.BoxSize);
            idx_min[jj] = kernel<mas>(rnorm, k[jj]);
        }

        for (int64_t ix=0; ix < Nkernel<mas>; ++ix)
            for (int64_t iy=0; iy < Nkernel<mas>; ++iy)
                for (int64_t iz=0; iz < Nkernel<mas>; ++iz)
                {
                    int64_t targ_idx = to_1d(ix+idx_min[0], iy+idx_min[1], iz+idx_min[2], Nm);
                    m[targ_idx] += k[0][ix] * k[1][iy] * k[2][iz];
                }
    }

    globals.time_mesh = get_time(start);

    return 0;
}

// instantiate
#define INSTANTIATE(mas, rsd) \
    template int mesh<mas, rsd> \
    (Globals &, const float *, const float *, int64_t, std::vector<float> &)

INSTANTIATE(MAS::NGP, RSD::None);
INSTANTIATE(MAS::NGP, RSD::x);
INSTANTIATE(MAS::NGP, RSD::y);
INSTANTIATE(MAS::NGP, RSD::z);
INSTANTIATE(MAS::CIC, RSD::None);
INSTANTIATE(MAS::CIC, RSD::x);
INSTANTIATE(MAS::CIC, RSD::y);
INSTANTIATE(MAS::CIC, RSD::z);
INSTANTIATE(MAS::TSC, RSD::None);
INSTANTIATE(MAS::TSC, RSD::x);
INSTANTIATE(MAS::TSC, RSD::y);
INSTANTIATE(MAS::TSC, RSD::z);
INSTANTIATE(MAS::PCS, RSD::None);
INSTANTIATE(MAS::PCS, RSD::x);
INSTANTIATE(MAS::PCS, RSD::y);
INSTANTIATE(MAS::PCS, RSD::z);
