/* Small helper script to compute FFTW wisdom
 * Command line arguments:
 *   [1]   output file name
 *   [2..] mesh sizes (integers)
 *         Pass as many as you want.
 */

#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cstring>

#include <fftw3.h>

int main(int argc, char **argv)
{
    const char *outbase = argv[1];

    fftwf_plan plans[argc];

    char compiler[32];
    #if defined(__ICC)
    std::sprintf(compiler, "icc%d", __ICC);
    #elif defined(__GNUC__)
    std::sprintf(compiler, "gcc%d%d", __GNUC__, __GNUC_MINOR__);
    #endif

    char fftwversion[32];
    char *dest = fftwversion;
    for (const char *c=fftwf_version; *c != 0; ++c, ++dest)
        *dest = *c;
    for (const char *c=fftwf_cc; *c != ' ' && *c != 0; ++c, ++dest)
        *dest = *c;
    *dest = 0;
    
    for (int ii=2; ii<argc; ++ii)
    {
        int64_t N = std::atol(argv[ii]);
        std::printf("Starting N=%ld ...\n", N);

        float *in = (float *)fftwf_alloc_real(N * N * N);
        std::complex<float> *out = (std::complex<float> *)fftwf_alloc_complex(N * N * (N/2+1));

        plans[ii] = fftwf_plan_dft_r2c_3d(N, N, N, in, (fftwf_complex *)out,
                                          FFTW_PATIENT | FFTW_DESTROY_INPUT);

        char buffer[512];
        std::sprintf(buffer, "%s_N%ld_%s_%s", outbase, N, compiler, fftwversion);
        fftwf_export_wisdom_to_filename(buffer);

        fftwf_free(in);
        fftwf_free(out);

        std::printf("... finished N=%ld\n", N);
    }

    return 0;
}
