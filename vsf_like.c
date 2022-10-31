/* Compute the log-likelihood for a single augmentation
 * relative to an observation.
 * Command line arguments (all integers):
 *   [1] N ... number of bins
 *   [2...N+2] ... the observed counts
 *   [N+3...2N+3] ... the simulated counts
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <gsl/gsl_sf_gamma.h>

int main(int argc, char **argv)
{
    char **c = argv+1;
    unsigned N = atoi(*(c++));
    unsigned xo[N], xs[N];
    for (unsigned ii=0; ii<N; ++ii)
        xo[ii] = atoi(*(c++));
    for (unsigned ii=0; ii<N; ++ii)
        xs[ii] = atoi(*(c++));
    assert((c-argv)==argc);

    // store in extended precision to minimize round-off during summation
    long double out = 0.0L;

    for (unsigned ii=0; ii<N; ++ii)
        out += (xo[ii]-xs[ii]) * M_LN2
               + gsl_sf_lnchoose(xo[ii]+xs[ii], xo[ii])
               - gsl_sf_lnchoose(2U*xo[ii], xo[ii]);

    printf("%.16Le\n", out);

    return 0;
}
