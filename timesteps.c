/* Quick script to figure out the FastPM time stepping.
 * Command line arguments:
 * [1] starting redshift (float)
 * [2] redshift to switch from logarithmic to linear steps (float)
 * [3] how many logarithmic steps to take (int)
 * [4] how many linear steps to take before reaching start of outputs (int)
 * [5] how many output times there are (int)
 * [6...] the output times (float...)
 *
 * Prints: comma separated list of scale factors for FastPM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int dbl_cmp (const void *a_, const void *b_)
{
    double a = *(double *)a_;
    double b = *(double *)b_;
    return (a>b) ? 1 : -1;
}

int main(int argc, char **argv)
{
    int ii;

    char **c = argv+1;

    double z_initial = atof(*(c++));
    double z_mid = atof(*(c++));
    int Nlog = atoi(*(c++));
    int Nlin = atoi(*(c++));
    int Nout = atoi(*(c++));

    double a_out[Nout];
    for (ii=0; ii<Nout; ++ii) a_out[ii] = atof(*(c++));

    if ((c - argv) != argc) return 1;

    qsort(a_out, Nout, sizeof(double), dbl_cmp);

    double a_initial = 1.0/(1.0+z_initial);
    double a_mid = 1.0/(1.0+z_mid);
    if (a_initial>a_mid) return 1;
    if (a_mid>a_out[0]) return 1;

    // this one includes a_mid
    double a_log[Nlog+1];
    for (ii=0; ii<=Nlog; ++ii)
        a_log[ii] = a_initial * pow(a_mid/a_initial, (double)ii/(double)Nlog);

    // this one does not include a_mid or a_out[0]
    double a_lin[Nlin];
    for (ii=0; ii<Nlin; ++ii)
        a_lin[ii] = a_mid + (a_out[0]-a_mid)*(double)(ii+1)/(double)(Nlin+1);

    // for the output, we adjust the last one slightly so the simulation actually reaches the end
    a_out[Nout-1] += 1e-3;

    // this is everything
    int Nall = Nlog+1+Nlin+Nout;
    double a_all[Nall];
    
    memcpy(a_all, a_log, (Nlog+1)*sizeof(double));
    memcpy(a_all+Nlog+1, a_lin, Nlin*sizeof(double));
    memcpy(a_all+Nlog+1+Nlin, a_out, Nout*sizeof(double));

    // output
    for (ii=0; ii<Nall; ++ii)
        printf("%.8f%s", a_all[ii], (ii==Nall-1) ? "" : ",");

    return 0;
}
