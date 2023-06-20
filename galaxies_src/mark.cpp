#include <cstdint>
#include <complex>
#include <memory>
#include <vector>
#include <cmath>

#include <mkl_dfti.h>
#include <gsl/gsl_math.h>

#include "globals.h"
#include "err.h"
#include "timing.h"

#include "mark.h"

#ifndef M_PIf32
#define M_PIf32 3.141592653589793238462643383279502884F
#endif

static inline int64_t wrap_idx (int64_t N, int64_t i)
// maps i=0..N-1 to i=0,..,N/2,N/2-1,..,1
// which is the fftw convention for the ordering of wavenumbers
{
    return (i<=N/2) ? i : N-i;
}

static inline float tophat (float kR)
{
    if (kR < 1e-5)
        return 1.0F - 0.1F * kR*kR + 0.0035714285714285713F * kR*kR*kR*kR;
    else
        return 3.0F / (kR*kR*kR) * ( std::sin(kR) - kR*std::cos(kR) );
}

static int filter (float BoxSize, int64_t N, std::complex<float> *m, float R)
{
    for (int64_t ix=0; ix<N; ++ix)
    {
        int64_t ixw = wrap_idx(N, ix);
        for (int64_t iy=0; iy<N; ++iy)
        {
            int64_t iyw = wrap_idx(N, iy);
            int64_t idx = (ix*N+iy) * (N/2+1);

            for (int64_t iz=0; iz<=N/2; ++iz, ++idx)
            {
                // in natural scaling
                float knorm = std::sqrt(ixw*ixw+iyw*iyw+iz*iz);

                float kR = 2.0F * M_PIf32 * R / BoxSize * knorm;

                m[idx] *= tophat(kR);
            }
        }
    }

    return 0;
}

static int filter (float BoxSize, int64_t N, float *m, float R)
{
    std::complex<float> *mcompl = (std::complex<float> *)std::aligned_alloc(32, N*N*N*2*sizeof(float));
    CHECK(!mcompl, return 1);

    MKL_LONG status;
    int status1;

    DFTI_DESCRIPTOR_HANDLE handle;

    MKL_LONG sizes[3]; for (int ii=0; ii<3; ++ii) sizes[ii] = N;
    MKL_LONG rstrides[4], cstrides[4];

    rstrides[3] = 1; rstrides[2] = N;     rstrides[1] = N*N;       rstrides[0] = 0;
    cstrides[3] = 1; cstrides[2] = N/2+1; cstrides[1] = N*(N/2+1); cstrides[0] = 0;

    status = DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 3, sizes);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    status = DftiSetValue(handle, DFTI_INPUT_STRIDES, rstrides);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    status = DftiSetValue(handle, DFTI_OUTPUT_STRIDES, cstrides);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    status = DftiSetValue(handle, DFTI_THREAD_LIMIT, 1);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    // this should correspond to the fftw convention
    status = DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    status = DftiCommitDescriptor(handle);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    // compute the forward transform
    status = DftiComputeForward(handle, m, mcompl);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    // apply tophat filter
    status1 = filter(BoxSize, N, mcompl, R); 
    CHECK(status1, return 1);

    // get back to real space -- note we need to reverse the strides now!
    status = DftiSetValue(handle, DFTI_INPUT_STRIDES, cstrides);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    status = DftiSetValue(handle, DFTI_OUTPUT_STRIDES, rstrides);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    status = DftiCommitDescriptor(handle);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    status = DftiComputeBackward(handle, mcompl, m);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    // clean up
    std::free(mcompl);
    DftiFreeDescriptor(&handle);

    // normalize properly (confirmed correct)
    // can also do this with the MKL scale setting
    for (int64_t ii=0; ii<N*N*N; ++ii)
        m[ii] /= N*N*N;

    return 0;
}

int mark (Globals &globals, std::vector<float> &m,
          float p, float delta_s, float R)
{
    int status;

    auto start = start_time();

    int64_t N = std::cbrt(m.size());
    CHECK(!(N*N*N == (int64_t)m.size()), return 1);

    float *mfiltered = (float *)std::aligned_alloc(32, N*N*N*sizeof(float));
    CHECK(!mfiltered, return 1);

    // compute sum for correct normalization, do this in double precision as no cost
    double sum = 0.0;
    for (int64_t ii=0; ii<N*N*N; ++ii)
        sum += (double)m[ii];
    double mean = sum / (N*N*N);

    for (int64_t ii=0; ii<N*N*N; ++ii)
    {
        // since we are doing something non-linear later, we need to subtract the mean here
        // However, we don't divide by it, since this is taken care of later by the power
        // spectrum computation which we don't want to touch
        m[ii] = m[ii] - mean;

        // here, however, we also need to divide, since we apply a non-linear function later to it
        mfiltered[ii] = m[ii] / mean;
    }

    // compute the filtered overdensity field
    status = filter(globals.BoxSize, N, mfiltered, R);

    // compute the mark, as well as its mean
    double sum_mark = 0.0;
    for (int64_t ii=0; ii<N*N*N; ++ii)
    {
        mfiltered[ii] = std::pow( (1.0F+delta_s)/(1.0F+delta_s+mfiltered[ii]), p);
        sum_mark += mfiltered[ii];
    }
    double mean_mark = sum_mark / (N*N*N);

    // and now apply the mark to our output
    for (int64_t ii=0; ii<N*N*N; ++ii)
        m[ii] *= mfiltered[ii] / mean_mark;

    // clean up
    std::free(mfiltered);

    globals.time_mark = get_time(start);

    return 0;
}
