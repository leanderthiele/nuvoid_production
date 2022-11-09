// Some code to produce QRNG samples from a Gaussian prior
// in conjunction with possible uniform distributions
// Command line arguments:
//  [1] index of the sample requested (>=0)
//  [2] dimensionality of gaussian part (integer)
//  [3] file that contains as first line the mean vector
//      and in the following lines the covariance matrix
//      [lines starting with # are ignored]
//      The covariance matrix/mean vector are d-1 dimensional
//  [4] dimensionality of uniform part (integer)
//  [5] file that contains in lines the min, max values of
//      the uniform part

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
#define SEED (1UL<<63) // can be anything but 0.5 recommended
                       // this value is in our uint64_t representation

// ---- global input variables ----
int d, gauss_d, uniform_d; // dimensionality

char *mu_cov_fname, *uniform_fname;

double cov[MAX_D*MAX_D]; // covariance matrix, gauss_d x gauss_d
double mu[MAX_D]; // mean vector, gauss_d

double Umin[MAX_D], Umax[MAX_D]; // for the uniform distribution

// ---- global work/output variables ----

// the Cholesky decomposition of the coviariance matrix
// s.t. chol chol^T == cov
double chol[MAX_D*MAX_D];

// the basis vector to traverse the unit hypercube
// mapped to the integers for stability
// d dimensional
uint64_t alpha[2*MAX_D];

// the Nth sample in the unit hypercube, d
double U01_sample[2*MAX_D];

// the Nth sample in the standard normal, gauss_d
double N01_sample[MAX_D];

// the Nth sample in the multivariate normal, gauss_d+1
// [last entry is from uniform]
double NmC_sample[2*MAX_D];

void read_mu_cov (void);
void read_uniform (void);
void prepare (void);
void compute_sample (uint64_t N);

int main (int argc, char **argv)
{
    char **c = argv+1;    
    uint64_t N = atoi(*(c++));
    gauss_d = atoi(*(c++));
    mu_cov_fname = *(c++);
    uniform_d = atoi(*(c++));
    uniform_fname = *(c++);

    assert(d<=MAX_D);
    assert(uniform_d<=MAX_D);
    assert((c-argv)==argc);

    d = gauss_d + uniform_d;

    read_mu_cov();
    read_uniform();
    prepare();

    #ifndef TEST
    compute_sample(N+1);

    for (int ii=0; ii<d; ++ii)
        printf("%.15f%s", NmC_sample[ii], (ii==(d-1))?"":",");
    #else
    double *x = malloc(N * d * sizeof(double));
    for (int ii=0; ii<N; ++ii)
    {
        compute_sample(ii+1);
        for (int jj=0; jj<d; ++jj)
            x[ii*d+jj] = NmC_sample[jj];
    }
    FILE *fp = fopen("sample_prior_test.bin", "wb");
    fwrite(x, sizeof(double), N*d, fp);
    fclose(fp);
    #endif

    return 0;
}

// ---- implementation ----

void read_mu_cov (void)
{
    char line_buffer[512];
    char fmt[gauss_d][512];
    for (int ii=0; ii<gauss_d; ++ii)
    {
        fmt[ii][0] = '\0'; // make strlen work
        for (int jj=0; jj<gauss_d; ++jj)
            sprintf(fmt[ii]+strlen(fmt[ii]), "%s", (ii==jj)?"%lf":"%*lf");
    }

    FILE *fp = fopen(mu_cov_fname, "r");
    int line_no = 0;
    double data_buffer[gauss_d];
    int read;
    while (fscanf(fp, "%[^\n]\n", line_buffer) != EOF)
    {
        if (line_buffer[0] == '#') continue;
        ++line_no;
        assert(line_no<=gauss_d+1);
        for (int ii=0; ii<gauss_d; ++ii)
        {
            read = sscanf(line_buffer, fmt[ii], data_buffer+ii);
            assert(read == 1);
        }
        if (line_no==1)
            for (int ii=0; ii<gauss_d; ++ii) mu[ii] = data_buffer[ii];
        else
            for (int ii=0; ii<gauss_d; ++ii) cov[(line_no-2)*gauss_d+ii] = data_buffer[ii];
    }
    assert(line_no==gauss_d+1);
    fclose(fp);
}

void read_uniform (void)
{
    char line_buffer[512];
    char fmt[2][32] = { "%lf%*lf", "%*lf%lf" };

    FILE *fp = fopen(uniform_fname, "r");
    int line_no = 0;
    int read;
    while (fscanf(fp, "%[^\n]\n", line_buffer) != EOF)
    {
        if (line_buffer[0] == '#') continue;
        ++line_no;
        assert(line_no<=uniform_d);
        read = sscanf(line_buffer, fmt[0], Umin+line_no-1);
        assert(read == 1);
        read = sscanf(line_buffer, fmt[1], Umax+line_no-1);
        assert(read == 1);
        assert(Umax[line_no-1] > Umin[line_no-1]);
    }
    assert(line_no==uniform_d);
    fclose(fp);
}

void compute_chol (void)
{
    gsl_matrix *C = gsl_matrix_alloc(gauss_d, gauss_d);
    for (int ii=0; ii<gauss_d; ++ii)
        for (int jj=0; jj<gauss_d; ++jj)
            gsl_matrix_set(C, ii, jj, cov[ii*gauss_d+jj]);

    gsl_linalg_cholesky_decomp1(C);

    for (int ii=0; ii<gauss_d*gauss_d; ++ii) chol[ii] = 0.0;
    for (int ii=0; ii<gauss_d; ++ii)
        for (int jj=0; jj<=ii; ++jj)
            chol[ii*gauss_d+jj] = gsl_matrix_get(C, ii, jj);

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
        alpha[ii] = (uint64_t)(((long double)(UINT64_MAX)+1.0L) * a);
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
        U01_sample[ii] = (long double)(SEED + N * alpha[ii])/((long double)(UINT64_MAX)+1.0L);
}

void compute_N01_sample (void)
{
    for (int ii=0; ii<gauss_d; ++ii)
        N01_sample[ii] = gsl_cdf_ugaussian_Pinv(U01_sample[ii]);
}

void compute_NmC_sample (void)
{
    // overwrites the vector Y, which is the offset
    for (int ii=0; ii<gauss_d; ++ii) NmC_sample[ii] = mu[ii];

    cblas_dgemv(CblasRowMajor, CblasNoTrans, gauss_d, gauss_d,
                /*alpha=*/1.0, chol, /*lda=*/gauss_d,
                /*X=*/N01_sample, /*incX=*/1,
                /*beta=*/1.0, /*Y=*/NmC_sample, /*incY=*/1);

    // add the uniform samples
    for (int ii=0; ii<uniform_d; ++ii)
        NmC_sample[gauss_d+ii] = Umin[ii] + U01_sample[gauss_d+ii] * (Umax[ii] - Umin[ii]);
}

void compute_sample (uint64_t N)
{
    compute_U01_sample(N);
    compute_N01_sample();
    compute_NmC_sample();
}
