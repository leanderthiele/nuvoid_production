/* To read from our hdf5 Rockstar catalogs */

#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <glob.h>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "read_hdf5.h"


static int read_spec (const char *s, const char *subs, float *out)
// searches the string s for the substring subs followed by a floating point number,
// which, if found, is written into out (else out remains untouched)
// Returns 1 if the substring was found, -1 if it was found but no floating point number
// identified afterwards, and 0 if substring was not found.
{// {{{
    char buffer[512];
    for (const char *c=s; *c != 0; ++c)
        if (!strncmp(c, subs, std::strlen(subs)))
        {
            int ii = 0;
            for (const char *c1=c+std::strlen(subs); *c1 != 0; ++c1, ++ii)
                if ( (*c1 >= '0' && *c1 <= '9')
                    || *c1 == '.' || *c1 == '+' || *c1 == '-'
                    || *c1 == 'e' || *c1 == 'E')
                    buffer[ii] = *c1;
            if (ii>0)
            {
                buffer[ii] = 0;
                *out = atof(buffer);
                return 1;
            }
            else
                return -1;
        }

    return 0;
}// }}}

static int read_header (hid_t file, float *BoxSize, float *z)
{// {{{
    herr_t status;

    char attr_name[16];
    char attr_val[512];
    int BoxSize_found=0, z_found=0;
    for (int ii=0; ii<16 && !(BoxSize_found && z_found); ++ii)
    {
        sprintf(attr_name, "ATTR%d", ii);
        status = H5LTget_attribute_string(file, "/Header", attr_name, attr_val);
        if (status<0) return 1;

        if (!BoxSize_found)
        {
            BoxSize_found = read_spec(attr_val, "#Box size: ", BoxSize);
            if (BoxSize_found<0) return 1;
        }
        if (!z_found)
        {
            float a;
            z_found = read_spec(attr_val, "#a = ", &a);
            if (z_found<0) return 1;
            if (z_found) *z = 1.0/a - 1.0;
        }
    }

    if (!(BoxSize_found && z_found)) return 1;
    return 0;
}// }}}

template<typename T, bool is_vec>
static int read_dset (hid_t file, const char *name, int64_t *N, int64_t Nused, T **out)
{// {{{
    hid_t hdf5_type;
    if (std::is_same<T, int64_t>::value) hdf5_type = H5T_NATIVE_LONG;
    else if (std::is_same<T, float>::value) hdf5_type = H5T_NATIVE_FLOAT;

    herr_t status;
    int rank;

    status = H5LTget_dataset_ndims(file, name, &rank);
    if (status<0) return 1;
    if ( (is_vec && rank != 2) || (!is_vec && rank != 1) ) return 1;

    hsize_t dims[rank];
    H5T_class_t class_id;
    size_t type_size;
    status = H5LTget_dataset_info(file, name, dims, &class_id, &type_size);
    if (status<0) return 1;
    if (is_vec && dims[1] != 3) return 1;
    if (sizeof(T) != type_size) return 1;

    int64_t thisN = dims[0];
    if (*N>0 && *N != thisN) return 1;
    *N = thisN;

    int64_t stride = (is_vec) ? 3 : 1 ;
    if (!Nused)
        *out = (T *)std::malloc(thisN * stride * sizeof(T));
    else
        *out = (T *)std::realloc(*out, (Nused + thisN) * stride * sizeof(T));

    T *dest = *out + Nused * stride;
    status = H5LTread_dataset(file, name, hdf5_type, dest);
    if (status<0) return 1;

    return 0;
}// }}}

template<Sec secondary, bool have_vel>
static int do_file (const char *fname,
                    float *BoxSize, float *z, int64_t *N,
                    float **M, float **pos,
                    [[maybe_unused]] float **vel,
                    [[maybe_unused]] float **sec)
{
    herr_t status; int status1;

    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file<0) return 1;

    status1 = read_header(file, BoxSize, z);
    if (status1) return 1;

    int64_t thisN = -1;
    status1 = read_dset<float, false>(file, "Mvir", &thisN, *N, M);
    if (status1) return 1;
    status1 = read_dset<float, true>(file, "Pos", &thisN, *N, pos);
    if (status1) return 1;

    if constexpr (have_vel)
    {
        status1 = read_dset<float, true>(file, "Vel", &thisN, *N, vel);
        if (status1) return 1;
    }

    if constexpr (secondary != Sec::None)
    {
        if constexpr (secondary == Sec::Conc)
        {
            if (!*N)
                *sec = (float *)std::malloc(thisN * sizeof(float));
            else
                *sec = (float *)std::realloc(*sec, (*N + thisN) * sizeof(float));

            float *Rvir = nullptr, *Rs = nullptr;
            status1 = read_dset<float, false>(file, "Rvir", &thisN, 0, &Rvir);
            if (status1) return 1;
            status1 = read_dset<float, false>(file, "Rs", &thisN, 0, &Rs);
            if (status1) return 1;

            float *conc_dest = *sec + *N;
            for (int64_t ii=0; ii<thisN; ++ii)
                conc_dest[ii] = Rvir[ii] / Rs[ii];

            if (Rvir) std::free(Rvir);
            if (Rs) std::free(Rs);
        }
        else if constexpr (secondary == Sec::TU)
        {
            status1 = read_dset<float, false>(file, "kin_to_pot", &thisN, *N, sec);
            if (status1) return 1;
        }
    }

    status = H5Fclose(file);
    if (status<0) return 1;

    *N += thisN;

    return 0;
}

template<Sec secondary, bool have_vel>
int read_hdf5 (const char *dirname,
               float *BoxSize, float *z, int64_t *N,
               float **M, float **pos,
               [[maybe_unused]] float **vel,
               [[maybe_unused]] float **sec)
// pass nullptr for conc if you don't need it
{
    int status;

    // initialize
    *N = 0;

    char buffer[512];
    glob_t glob_result;

    sprintf(buffer, "%s/out_*[0-9]_hosts.hdf5", dirname);
    status = glob(buffer, GLOB_TILDE_CHECK, NULL, &glob_result);
    if (status) return 1;
    if (glob_result.gl_pathc != 1) { fprintf(stderr, "Found !=1 hosts files."); return 1; }
    status = do_file<secondary, have_vel>
        (glob_result.gl_pathv[0], BoxSize, z, N, M, pos, vel, sec);
    if (status) return 1;
    globfree(&glob_result);

    return 0;
}

// instantiate
#define INSTANTIATE(secondary, have_vel) \
    template int read_hdf5<secondary, have_vel> \
    (const char *, float *, float *, int64_t *, \
     float **, float **, float **, float **)

INSTANTIATE(Sec::None, true);
INSTANTIATE(Sec::None, false);
INSTANTIATE(Sec::Conc, true);
INSTANTIATE(Sec::Conc, false);
INSTANTIATE(Sec::TU, true);
INSTANTIATE(Sec::TU, false);
