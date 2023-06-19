/* some nfw utilities */

#ifndef NFW_H
#define NFW_H

#include <cmath>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_dilog.h>
#include <gsl/gsl_sf_lambert.h>

namespace NFW {

template<typename T>
T g (T x)
// the canonical NFW function for enclosed mass
{
    return std::log1p(x) - x/ (1.0 + x);
}

double q (double c, double p)
// the solution from [1805.09550]. p is in [0, 1] and parameterizes the enclosed mass.
// Returns radius in units of the cutoff radius, in [0, 1]
{
    double M1 = g(c);
    return -1.0/c * ( 1.0 + 1.0/gsl_sf_lambert_W0(-std::exp(-1.0-p*M1)) );
}

double zeta (double x)
// zeta(x) = \int_x^infty dy g(y) / [ y^3 * (1+y)^2 ]
// Confirmed with Mathematica that this function is correct
// Divergence at zero is extremely weak
// Called with values in the interval [0, cmax] ~ [0, 15]
{
    const static double eps = 1e-8; // do not evaluate below this

    if (x < eps)
        return zeta(eps);

    return 0.5/gsl_pow_2(x * (1.0+x)) *
           ( x * ( -1.0+x*(-9.0-7.0*x+gsl_pow_2(M_PI * (1.0+x))) )
            + gsl_pow_4(x) * std::log1p(1.0/x) + std::log1p(x)
            + x * ( -x*(1.0+2.0*x)*std::log(x)
                    + std::log1p(x) * (-2.0-4.0*x*(2.0+x)+3.0*x*gsl_pow_2(1.0+x)*std::log1p(x)) )
            + 6.0 * gsl_pow_2(x * (1.0+x)) * gsl_sf_dilog(-x)
           );
}

} // namespace NFW

#endif // NFW_H
