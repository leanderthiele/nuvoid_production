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
    {
        gsl_sf_result r1, r2;
        gsl_sf_lnchoose_e(xo[ii]+xs[ii], xo[ii], &r1);
        gsl_sf_lnchoose_e(2U*xo[ii], xo[ii], &r2);

        printf("val=%.16e err=%.16e\n", r1.val, r1.err);
        printf("val=%.16e err=%.16e\n", r2.val, r2.err);
        printf("diff=%.16e err=%.16e\n", fabs(r1.val-r2.val), fmax(r1.err, r2.err));

        out += (xo[ii]-xs[ii]) * M_LN2 + r1.val - r2.val;
    }

    printf("%.16Le\n", out);

    return 0;
}
