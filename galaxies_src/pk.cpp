#include <complex>
#include <memory>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <mkl_dfti.h>
#include <gsl/gsl_math.h>

#include "enums.h"
#include "globals.h"
#include "err.h"
#include "timing.h"

#ifndef M_PIf32
#define M_PIf32 3.141592653589793238462643383279502884F
#endif

static int fft (int64_t N, float *m, std::complex<float> *mcompl)
{
    MKL_LONG status;

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

    status = DftiComputeForward(handle, m, mcompl);
    CHECK(status!=DFTI_NO_ERROR, return 1);

    DftiFreeDescriptor(&handle);

    return 0;
}

static inline int64_t wrap_idx (int64_t N, int64_t i)
// maps i=0..N-1 to i=0,..,N/2,N/2-1,..,1
// which is the fftw convention for the ordering of wavenumbers
{
    return (i<=N/2) ? i : N-i;
}

static inline float invsinc (int64_t N, int64_t i)
{
    if (!i)
        return 1.0F;
    else
    {   
        float x = M_PIf32 * (float)i / (float)N;
        return x / std::sin(x);
    }
}


template<MAS mas>
static inline float invWk (int64_t N, int64_t ix, int64_t iy, int64_t iz)
// inverse of fourier space window function
{
    float y = invsinc(N, ix) * invsinc(N, iy) * invsinc(N, iz);
    if      constexpr (mas==MAS::NGP) return y;
    else if constexpr (mas==MAS::CIC) return y*y;
    else if constexpr (mas==MAS::TSC) return y*y*y;
    else if constexpr (mas==MAS::PCS) return y*y*y*y;
}

template<MAS mas>
static float C1 (int64_t N, int64_t i)
// direction averaged approximation to the fourier space window function
{
    float q = (float)(i+1) * M_PIf32 / (float)N;
    if      constexpr (mas == MAS::NGP) return 1.0F;
    else if constexpr (mas == MAS::CIC) return 1.0F - 2.0F/3.0F * gsl_pow_2(std::sin(q));
    else if constexpr (mas == MAS::TSC) return 1.0F - gsl_pow_2(std::sin(q))
                                                    + 2.0F/15.0F * gsl_pow_4(std::sin(q));
    else if constexpr (mas == MAS::PCS) return 1.0F - 4.0F/3.0F * gsl_pow_2(std::sin(q))
                                                    + 2.0F/5.0F * gsl_pow_4(std::sin(q))
                                                    - 4.0F/315.0F * gsl_pow_6(std::sin(q));
}


template<MAS mas, RSD rsd>
static void bin_pk (float BoxSize, int64_t Ngals, int64_t N, std::complex<float> *mcompl, float kmax,
                    std::vector<float> &k, std::vector<float> &Pk, std::vector<int64_t> &Nk)
// choose k-bins as [i+0.5, i+1.5) * 2pi/L, go up to 0.5*k_nyq
{
    // maximum k bin index
    const int64_t imax = std::min(N/2-1, (int64_t)std::ceil(BoxSize*kmax/(2.0F*M_PIf32)-1.0F));

    k.resize(imax+1);
    const int64_t Nbins = k.size();

    for (int64_t ii=0; ii<Nbins; ++ii)
        k[ii] = (ii+1.0F) * 2.0F * M_PIf32 / BoxSize;

    Pk.resize((rsd==RSD::None) ? Nbins : 3 * Nbins, 0.0F);
    Nk.resize(Nbins, 0);

    // NOTE this is sufficiently optimized. For reasonable kmax this runs in negligible time
    //      compared to the FFT
    for (int64_t ix=0; ix<N; ++ix)
    {
        int64_t ixw = wrap_idx(N, ix);
        for (int64_t iy=0; iy<N; ++iy)
        {
            int64_t iyw = wrap_idx(N, iy);
            
            // this index points into the complex array, it is incremented below
            int64_t idx = (ix*N+iy) * (N/2+1);
            
            for (int64_t iz=0; iz<=N/2; ++iz, ++idx)
            {
                // in the natural scaling
                float knorm = std::sqrt(ixw*ixw+iyw*iyw+iz*iz);

                int64_t jj = std::floor(knorm - 0.5);
                if      (jj<0)    continue;
                else if (jj>imax) break; // the z wavenumber only increases afterwards

// The following two lines can be used to compensate for the window function,
// in which case the C1 factor in the shot noise correction should not be used.
// However, the compensation is quite expensive and as long as we stick to a
// consistent kernel it shouldn't matter much.
//                float invW = invWk<mas>(N, ixw, iyw, iz);
//                Pk[jj] += std::norm(mcompl[idx]) * invW * invW;

                float this_pk = std::norm(mcompl[idx]);

                Pk[jj] += this_pk;

                if constexpr (rsd != RSD::None)
                {
                    // the k along the line of sight
                    int64_t klos = (rsd==RSD::x) ? ixw : (rsd==RSD::y) ? iyw : iz;
                    float x = klos / knorm; // note knorm==0 not possible
                    
                    // quadrupole
                    Pk[1*Nbins+jj] += this_pk * (-2.5F + 7.5F*x*x);

                    // hexadecapole
                    Pk[2*Nbins+jj] += this_pk * (3.375 - 33.75*x*x + 39.375*x*x*x*x);
                }

                ++Nk[jj];
            }
        }
    }

    for (int64_t ii=0; ii<( (rsd==RSD::None) ? 1 : 3 ); ++ii)
        for (int64_t jj=0; jj<Nbins; ++jj)
            // have confirmed shot noise correction using randoms
            Pk[ii*Nbins+jj] = gsl_pow_3(BoxSize)/(Ngals*Ngals)
                              * ( Pk[ii*Nbins+jj]/Nk[jj]
                                 - ( (ii==0) ? Ngals*C1<mas>(N, jj) : 0 ) );

    for (int64_t ii=0; ii<Nbins; ++ii)
        Nk[ii] *= 2;
}

template<MAS mas, RSD rsd>
int pk (Globals &globals, const std::vector<float> &m,
        float kmax,
        std::vector<float> &k, std::vector<float> &Pk, std::vector<int64_t> &Nk)
// if RSD is on, the returned power spectrum contains three blocks, each of length
// k.size(), with the monopole, quadrupole, and hexadecapole
{
    auto start = start_time();
    
    int64_t N = std::cbrt(m.size());
    CHECK(!(N*N*N == (int64_t)m.size()), return 1);

    float *mreal = (float *)std::aligned_alloc(32, N*N*N*sizeof(float));
    CHECK(!mreal, return 1);

    std::complex<float> *mcompl = (std::complex<float> *)std::aligned_alloc(32, N*N*N*2*sizeof(float));
    CHECK(!mcompl, return 1);

    for (int64_t ii=0; ii<N*N*N; ++ii)
        mreal[ii] = m[ii];

    auto start1 = start_time();
    int status = fft(N, mreal, mcompl);
    CHECK(status, return 1);
    globals.time_fft = get_time(start1);

    auto start2 = start_time();
    bin_pk<mas, rsd>(globals.BoxSize, globals.Ngals, N, mcompl, kmax, k, Pk, Nk);
    globals.time_bin = get_time(start2);

    std::free(mreal);
    std::free(mcompl);

    globals.time_pk = get_time(start);

    return 0;
}

// instantiate
#define INSTANTIATE(mas, rsd) \
    template int pk<mas, rsd> \
    (Globals &, const std::vector<float> &, \
     float, \
     std::vector<float> &, std::vector<float> &, std::vector<int64_t> &)

INSTANTIATE(MAS::NGP, RSD::None);
INSTANTIATE(MAS::NGP, RSD::x);
INSTANTIATE(MAS::NGP, RSD::y);
INSTANTIATE(MAS::NGP, RSD::z);
INSTANTIATE(MAS::CIC, RSD::None);
INSTANTIATE(MAS::CIC, RSD::x);
INSTANTIATE(MAS::CIC, RSD::y);
INSTANTIATE(MAS::CIC, RSD::z);
INSTANTIATE(MAS::TSC, RSD::None);
INSTANTIATE(MAS::TSC, RSD::x);
INSTANTIATE(MAS::TSC, RSD::y);
INSTANTIATE(MAS::TSC, RSD::z);
INSTANTIATE(MAS::PCS, RSD::None);
INSTANTIATE(MAS::PCS, RSD::x);
INSTANTIATE(MAS::PCS, RSD::y);
INSTANTIATE(MAS::PCS, RSD::z);
