#include <cstring>
#include <stdexcept>
#include <string>
#include <glob.h>

#include <bigfile.h>

#include "have_if.h"
#include "enums.h"
#include "err.h"

#include "read_bigfile.h"

static int read_spec (const char *s, const char *subs, float *out)
// searches the string s for the substring subs followed by a floating point number,
// which, if found, is written into out (else out remains untouched)
// Returns 1 if the substring was found, -1 if it was found but no floating point number
// identified afterwards, and 0 if substring was not found.
{// {{{
    char buffer[512];
    for (const char *c=s; *c != 0; ++c)
        if (!std::strncmp(c, subs, std::strlen(subs)))
        {
            int ii = 0;
            for (const char *c1=c+std::strlen(subs); *c1 != 0; ++c1, ++ii)
                if ( (*c1 >= '0' && *c1 <= '9')
                    || *c1 == '.' || *c1 == '+' || *c1 == '-'
                    || *c1 == 'e' || *c1 == 'E')
                    buffer[ii] = *c1;
                else break;

            if (ii>0)
            {
                buffer[ii] = 0;
                *out = std::atof(buffer);
                return 1;
            }
            else
                return -1;
        }

    return 0;
}// }}}

template<Cat cat_type>
static int read_header (BigFile *bf, float *BoxSize, float *z,
                        [[maybe_unused]] void *other);

template<>
int read_header<Cat::Rockstar> (BigFile *bf, float *BoxSize, float *z,
                                [[maybe_unused]] void *other)
{
    int status;

    BigBlock header;
    status = big_file_open_block(bf, &header, "Header");
    CHECK(status, return 1);

    char attr_name[16];
    int BoxSize_found=0, z_found=0;
    for (int ii=0; ii<16 && !(BoxSize_found && z_found); ++ii)
    {
        std::sprintf(attr_name, "ATTR%d", ii);
        BigAttr *attr = big_block_lookup_attr(&header, attr_name);
        CHECK(!attr, return 1);

        if (!BoxSize_found)
        {
            BoxSize_found = read_spec(attr->data, "#Box size: ", BoxSize);
            CHECK(BoxSize_found<0, return 1);
        }

        if (!z_found)
        {
            float a;
            z_found = read_spec(attr->data, "#a = ", &a);
            CHECK(z_found<0, return 1);
            if (z_found) *z = 1.0/a - 1.0;
        }
    }

    CHECK(!(BoxSize_found && z_found), return 1);

    status = big_block_close(&header);
    CHECK(status, return 1);

    return 0;
}

static int read_header_fastpm (BigFile *bf, float *BoxSize, float *z, void *other)
{
    int status;

    BigBlock header;
    status = big_file_open_block(bf, &header, "Header");
    CHECK(status, return 1);

    double time;
    status = big_block_get_attr(&header, "Time", &time, "f8", 1);
    CHECK(status, return 1);
    *z = 1.0/time - 1.0;

    double dbl_boxsize;
    status = big_block_get_attr(&header, "BoxSize", &dbl_boxsize, "f8", 1);
    CHECK(status, return 1);
    *BoxSize = dbl_boxsize;

    double mass_table[6];
    status = big_block_get_attr(&header, "MassTable", mass_table, "f8", 6);
    CHECK(status, return 1);
    *(double *)other = mass_table[1] * 1e10;

    status = big_block_close(&header);
    CHECK(status, return 1);

    return 0;
}

template<>
int read_header<Cat::FOF> (BigFile *bf, float *BoxSize, float *z, void *other)
{
    return read_header_fastpm(bf, BoxSize, z, other);
}

template<>
int read_header<Cat::RFOF> (BigFile *bf, float *BoxSize, float *z, void *other)
{
    return read_header_fastpm(bf, BoxSize, z, other);
}

template<typename T, bool is_vec>
static int read_dset (BigFile *bf, const char *name, int64_t *N, int64_t Nused, T **out)
{
    int status;

    char bf_type[8];
    if      constexpr (std::is_same<T, int64_t>::value) std::sprintf(bf_type, "i8");
    else if constexpr (std::is_same<T, int32_t>::value) std::sprintf(bf_type, "i4");
    else if constexpr (std::is_same<T, float>::value)   std::sprintf(bf_type, "f4");

    BigBlock bb;
    status = big_file_open_block(bf, &bb, name);
    CHECK(status, return 1);

    BigBlockPtr ptr;
    status = big_block_seek(&bb, &ptr, 0);
    CHECK(status, return 1);

    CHECK((is_vec && bb.nmemb != 3) || (!is_vec && bb.nmemb != 1), return 1);

    int64_t thisN = bb.size;
    CHECK(*N>0 && *N != thisN, return 1);
    *N = thisN;

    int64_t stride = (is_vec) ? 3 : 1;
    if (!Nused)
        *out = (T *)std::malloc(thisN * stride * sizeof(T));
    else
        *out = (T* )std::realloc(*out, (Nused + thisN) * stride * sizeof(T));

    void *dest = *out + Nused * stride;

    size_t dims[2];
    dims[0] = thisN;
    dims[1] = 3;

    BigArray ba;
    status = big_array_init(&ba, dest, bf_type, (is_vec) ? 2 : 1, dims, /*strides=*/NULL); 
    CHECK(status, return 1);

    status = big_block_read(&bb, &ptr, &ba);
    CHECK(status, return 1);

    status = big_block_close(&bb);
    CHECK(status, return 1);

    return 0;
}

template<Cat cat_type, Sec secondary, bool have_vel>
static int do_file (const char *fname,
                    float *BoxSize, float *z, int64_t *N,
                    float **M, float **pos,
                    [[maybe_unused]] float **vel,
                    [[maybe_unused]] float **sec)
{
    int status;

    BigFile bf;
    status = big_file_open(&bf, fname);
    CHECK(status, return 1);

    constexpr bool is_fof_cat = cat_type==Cat::FOF || cat_type==Cat::RFOF;
    HAVE_IF(is_fof_cat, double) unit_mass;
    HAVE_IF(is_fof_cat, std::string) fof_root;

    if constexpr (cat_type==Cat::FOF)
        fof_root = "LL-0.200/";
    else if constexpr (cat_type==Cat::RFOF)
        fof_root = "RFOF/";

    if constexpr (cat_type == Cat::Rockstar)
        status = read_header<Cat::Rockstar>(&bf, BoxSize, z, nullptr);
    else if constexpr (is_fof_cat)
        status = read_header<cat_type>(&bf, BoxSize, z, &unit_mass);
    CHECK(status, return 1);

    int64_t thisN = -1;
    if constexpr (cat_type == Cat::Rockstar)
    {
        status = read_dset<float, false>(&bf, "Mvir", &thisN, *N, M);
        CHECK(status, return 1);
        status = read_dset<float, true>(&bf, "Pos", &thisN, *N, pos);
        CHECK(status, return 1);
    }
    else if constexpr (is_fof_cat)
    {
        status = read_dset<float, true>(&bf, (fof_root+"Position").c_str(), &thisN, *N, pos);
        CHECK(status, return 1);

        // the FastPM FOF output is a bit annoying but whatever
        if (!*N)
            *M = (float *)std::malloc(thisN * sizeof(float));
        else
            *M = (float *)std::realloc(*M, (*N + thisN) * sizeof(float));

        int32_t *length = nullptr;
        status = read_dset<int32_t, false>(&bf, (fof_root+"Length").c_str(), &thisN, 0, &length);

        float *M_dest = *M + *N;
        for (int64_t ii=0; ii<thisN; ++ii)
            M_dest[ii] = (double)length[ii] * unit_mass;

        if (length) std::free(length);
    }
    else
        CHECK(1, return 1);

    if constexpr (have_vel)
    {
        if constexpr (cat_type == Cat::Rockstar)
            status = read_dset<float, true>(&bf, "Vel", &thisN, *N, vel);
        else if constexpr (is_fof_cat)
            status = read_dset<float, true>(&bf, (fof_root+"Velocity").c_str(), &thisN, *N, vel);
        CHECK(status, return 1);
    }

    if constexpr (secondary != Sec::None)
    {
        if constexpr (secondary == Sec::Conc)
        {
            static_assert(!is_fof_cat);

            if (!*N)
                *sec = (float *)std::malloc(thisN * sizeof(float));
            else
                *sec = (float *)std::realloc(*sec, (*N + thisN) * sizeof(float));

            float *Rvir = nullptr, *Rs = nullptr;
            status = read_dset<float, false>(&bf, "Rvir", &thisN, 0, &Rvir);
            CHECK(status, return 1);
            status = read_dset<float, false>(&bf, "Rs", &thisN, 0, &Rs);
            CHECK(status, return 1);

            float *conc_dest = *sec + *N;
            for (int64_t ii=0; ii<thisN; ++ii)
                conc_dest[ii] = Rvir[ii] / Rs[ii];

            if (Rvir) std::free(Rvir);
            if (Rs) std::free(Rs);
        }
        else if constexpr (secondary == Sec::TU)
        {
            static_assert(!is_fof_cat);

            status = read_dset<float, false>(&bf, "kin_to_pot", &thisN, *N, sec);
            CHECK(status, return 1);
        }
    }

    status = big_file_close(&bf);
    CHECK(status, return 1);

    *N += thisN;

    return 0;
}

template<Cat cat_type, Sec secondary, bool have_vel>
int read_bigfile (const char *dirname,
                  float *BoxSize, float *z, int64_t *N,
                  float **M, float **pos,
                  [[maybe_unused]] float **vel,
                  [[maybe_unused]] float **sec)
{
    int status;

    // initialize
    *N = 0;

    if constexpr (cat_type==Cat::FOF || cat_type==Cat::RFOF)
    {
        status = do_file<cat_type, secondary, have_vel>
            (dirname, BoxSize, z, N, M, pos, vel, sec);
        CHECK(status, return 1);
    }
    else if constexpr (cat_type == Cat::Rockstar)
    {
        char buffer[512];
        glob_t glob_result;

        sprintf(buffer, "%s/out_*[0-9]_hosts.bf", dirname);
        status = glob(buffer, GLOB_TILDE_CHECK, NULL, &glob_result);

        // FIXME
        if (status)
            fprintf(stderr, "%s\n", buffer);

        CHECK(status, return 1);
        CHECK(glob_result.gl_pathc != 1, return 1);
        status = do_file<cat_type, secondary, have_vel>
            (glob_result.gl_pathv[0], BoxSize, z, N, M, pos, vel, sec);
        CHECK(status, return 1);
        globfree(&glob_result);
    }
    else
        CHECK(1, return 1);

    return 0;
}

#define INSTANTIATE(cat_type, secondary, have_vel) \
    template int read_bigfile<cat_type, secondary, have_vel> \
    (const char *, float *, float *, int64_t *, \
     float **, float **, float **, float **)

INSTANTIATE(Cat::FOF, Sec::None, false);
INSTANTIATE(Cat::FOF, Sec::None, true);
INSTANTIATE(Cat::RFOF, Sec::None, false);
INSTANTIATE(Cat::RFOF, Sec::None, true);
INSTANTIATE(Cat::Rockstar, Sec::None, false);
INSTANTIATE(Cat::Rockstar, Sec::None, true);
INSTANTIATE(Cat::Rockstar, Sec::Conc, false);
INSTANTIATE(Cat::Rockstar, Sec::Conc, true);
INSTANTIATE(Cat::Rockstar, Sec::TU, false);
INSTANTIATE(Cat::Rockstar, Sec::TU, true);
