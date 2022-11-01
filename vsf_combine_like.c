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
#include <math.h>
#include <float.h>
#include <assert.h>

int main(int argc, char **argv)
{
    char **c = argv+1;
    int N = atoi(*(c++));
    long double loglikes[N];
    for (int ii=0; ii<N; ++ii)
        sscanf(*(c++), "%Lf", loglikes+ii);

    assert((c-argv)==argc);

    // take out the smallest value to avoid bad things from happening when exponentiating
    long double min = LDBL_MAX;
    for (int ii=0; ii<N; ++ii)
        if (min>loglikes[ii]) min = loglikes[ii];

    long double out = 0.0L;
    for (int ii=0; ii<N; ++ii)
        out += expl(loglikes[ii]-min);

    out = min + logl(out);

    printf("%.16Le\n", out);

    return 0;
}
