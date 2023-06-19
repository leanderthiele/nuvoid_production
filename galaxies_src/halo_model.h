/* Some small utilities for halo model calculations */

#ifndef HALO_MODEL_H
#define HALO_MODEL_H

#include <cmath>

#include "enums.h"

#define CUBE(x) ((x)*(x)*(x))

#ifndef M_PIf32
#define M_PIf32 3.141592653589793238462643383279502884F
#endif

static float Duffy08 (float M, float z, HaloDef mdef)
// concentration model of Duffy+2008
{
    const static float params[] = { 5.71, -0.087, -0.47, // 200c
                                   7.85, -0.081, -0.71, // vir
                                  10.14, -0.081, -1.01, // 200m
                            };
    auto p = params + 3 * (int)mdef;
    return p[0] * std::pow(M/1e12, p[1]) * std::pow(1.0+z, p[2]);
}

static float BryanNorman98 (float Omega_m, float z)
// virial fitting formula
{
    float x = Omega_m * CUBE(1.0F+z) - 1.0F;
    return 18.0F * M_PIf32 * M_PIf32 + 82.0F*x - 39.0F*x*x;
}

static float rho_c (float Omega_m, float z)
{
    return 2.7754e11F * ((1.0F-Omega_m) + Omega_m * CUBE(1.0F+z));
}

static float rho_m (float Omega_m, float z)
{
    return 2.7754e11F * Omega_m * CUBE(1.0F+z);
}

static float density_threshold (float Omega_m, float z, HaloDef mdef)
{
    switch (mdef)
    {
        case HaloDef::c : return 200.0 * rho_c(Omega_m, z);
        case HaloDef::v : return BryanNorman98(Omega_m, z) * rho_c(Omega_m, z);
        case HaloDef::m : return 200.0 * rho_m(Omega_m, z);
        default : return std::nanf("0");
    }
}

static float RofM (float M, float Omega_m, float z, HaloDef mdef)
{
    float dt = density_threshold(Omega_m, z, mdef);
    return std::cbrt(3.0F * M / 4.0F / M_PIf32 / dt);
}

static float VofM (float M, float Omega_m, float z, HaloDef mdef)
// returns newtonian virial velocity in km/s
{
    // number is G_newton in (Mpc/Msun)*(km/s)^2
    const static float a = std::sqrt(4.30091e-9) * std::pow(4.0*M_PI/3.0, 1.0/6.0);
    float dt = density_threshold(Omega_m, z, mdef);
    return a * std::cbrt(M) * std::pow(dt, 1.0/6.0);
}

#endif // HALO_MODEL_H
