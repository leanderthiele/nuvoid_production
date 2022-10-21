// Some code to produce QRNG samples from a Gaussian prior
// Command line arguments:
//  [1] dimensionality (integer)
//  [2] index of the sample requested (>=0)
//  [3] file that contains as first line the mean vector
//      and in the following lines the covariance matrix

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_cblas.h>

#define MAX_D 16

// ---- global input variables ----
int d; // dimensionality

char *mu_cov_fname;

double cov[MAX_D*MAX_D]; // covariance matrix, d x d
double mu[MAX_D]; // mean vector, d

// ---- global work/output variables ----

// the Cholesky decomposition of the coviariance matrix
// s.t. chol chol^T == cov
double chol[MAX_D*MAX_D];

// the basis vector to traverse the unit hypercube
// mapped to the integers for stability
uint64_t alpha[MAX_D];

// the Nth sample in the unit hypercube
double U01_sample[MAX_D];

// the Nth sample in the standard normal
double N01_sample[MAX_D];

// the Nth sample in the multivariate normal
double NmC_sample[MAX_D];

void read_mu_cov (void);
void prepare (void);
void compute_sample (uint64_t N);

int main (int argc, char **argv)
{
    char **c = argv+1;    
    d = atoi(*(c++));
    uint64_t N = atoi(*(c++));
    mu_cov_fname = *(c++);
    assert((c-argv)==argc);

    read_mu_cov();
    prepare();
    compute_sample(N+1);

    for (int ii=0; ii<d; ++ii)
        printf("%.15f%s", NmC_sample[ii], (ii==(d-1))?"":",");

    return 0;
}

// ---- implementation ----

void read_mu_cov (void)
{
    char line_buffer[512];
    char fmt[d][512];
    for (int ii=0; ii<d; ++ii)
    {
        fmt[ii][0] = '\0'; // make strlen work
        for (int jj=0; jj<d; ++jj)
            sprintf(fmt[ii]+strlen(fmt[ii]), "%s", (ii==jj)?"%lf":"%*lf");
    }

    FILE *fp = fopen(mu_cov_fname, "r");
    int line_no = 0;
    double data_buffer[d];
    while (fscanf(fp, "%[^\n]\n", line_buffer) != EOF)
    {
        ++line_no;
        assert(line_no<=d+1);
        for (int ii=0; ii<d; ++ii)
            sscanf(line_buffer, fmt[ii], data_buffer+ii);
        if (line_no==1)
            for (int ii=0; ii<d; ++ii) mu[ii] = data_buffer[ii];
        else
            for (int ii=0; ii<d; ++ii) cov[(line_no-2)*d+ii] = data_buffer[ii];
    }
    assert(line_no==d+1);
    fclose(fp);
}

void compute_chol (void)
{
    gsl_matrix *C = gsl_matrix_alloc(d, d);
    for (int ii=0; ii<d; ++ii)
        for (int jj=0; jj<d; ++jj)
            gsl_matrix_set(C, ii, jj, cov[ii*d+jj]);

    gsl_linalg_cholesky_decomp1(C);

    for (int ii=0; ii<d*d; ++ii) chol[ii] = 0.0;
    for (int ii=0; ii<d; ++ii)
        for (int jj=0; jj<=ii; ++jj)
            chol[ii*d+jj] = gsl_matrix_get(C, ii, jj);

    gsl_matrix_free(C);
}

void compute_alpha (void)
{
    long double phid = 2.0L;
    // find solution to phid^(d+1) = phid + 1,
    // the "generalized golden ratio"
    for (int ii=0; ii<32; ++ii)
        phid = powl(1.0L+phid, 1.0L/(1.0L+d));

    assert(fabsl(powl(phid, 1.0L+d)-(1.0L+phid))<1e-15L);

    for (int ii=0; ii<d; ++ii)
    {
        long double a = 1.0L / powl(phid, 1.0L+ii);
        assert(a<1.0L);
        alpha[ii] = (uint64_t)((1UL<<63) * a)<<1;
    }
}

void prepare (void)
{
    compute_chol();
    compute_alpha();
}


void compute_U01_sample (uint64_t N)
{
    assert(N>0); // otherwise the Pinv call below will produce NaN
    for (int ii=0; ii<d; ++ii)
        U01_sample[ii] = (long double)((N * alpha[ii])>>1)/(long double)(1UL<<63);
}

void compute_N01_sample (void)
{
    for (int ii=0; ii<d; ++ii)
        N01_sample[ii] = gsl_cdf_ugaussian_Pinv(U01_sample[ii]);
}

void compute_NmC_sample (void)
{
    // overwrites the vector Y, which is the offset
    for (int ii=0; ii<d; ++ii) NmC_sample[ii] = mu[ii];

    cblas_dgemv(CblasRowMajor, CblasNoTrans, d, d,
                /*alpha=*/1.0, chol, /*lda=*/d,
                /*X=*/N01_sample, /*incX=*/1,
                /*beta=*/1.0, /*Y=*/NmC_sample, /*incY=*/1);
}

void compute_sample (uint64_t N)
{
    compute_U01_sample(N);
    compute_N01_sample();
    compute_NmC_sample();
}
