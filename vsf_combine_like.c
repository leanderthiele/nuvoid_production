/* Combine the log-likelihoods from different augmentations into
 * the total log-likelihood for the specific HOD.
 * Command line arguments:
 *   [1] N ... number of augmentations
 *   [2..] the individual log-likelihoods
 * 
 * Prints out the combined log-likelihood.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include <gsl/gsl_statistics_long_double.h>

int main(int argc, char **argv)
{
    char **c = argv+1;
    int N = atoi(*(c++));
    long double loglikes[N];
    for (int ii=0; ii<N; ++ii)
        sscanf(*(c++), "%Lf", loglikes+ii);

    assert((c-argv)==argc);

    // take out the median value to avoid bad things from happening when exponentiating
    // we do this with a separate buffer to keep the ordering
    // (not necessary at the moment but it's ok)
    long double buffer[N];
    memcpy(buffer, loglikes, N*sizeof(long double));
    long double median = gsl_stats_long_double_median(buffer, 1, N);

    long double out = 0.0L;
    for (int ii=0; ii<N; ++ii)
        out += expl(loglikes[ii]-median);

    // take out dependence on N in the above sum
    out = median + logl(out/N);

    printf("%.16Le\n", out);

    return 0;
}
