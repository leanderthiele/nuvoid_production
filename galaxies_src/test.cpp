#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "enums.h"
#include "globals.h"
#include "hod_params.h"
#include "galaxies.h"
#include "power.h"

// for quick testing

template<typename T>
static void save_vec (const char *fname, const std::vector<T> &x)
{
    auto f = std::fopen(fname, "w");
    std::fwrite(x.data(), sizeof(T), x.size(), f);
    std::fclose(f);
}

constexpr auto secondary = Sec::Conc;

// settings for power spectrum estimator
constexpr int64_t Nmesh = 512;
constexpr auto mas = MAS::PCS;


int main(int argc, char **argv)
{
    int status = 0;

    char *inbase = argv[1];
    char *outbase = argv[2];

    HODParams<secondary != Sec::None, false> hod_params;
    hod_params.log_Mmin = 13.03;
    hod_params.sigma_logM = 0.38;
    hod_params.log_M0 = 13.27;
    hod_params.log_M1 = 14.08;
    hod_params.alpha = 0.76;

    if constexpr (secondary != Sec::None)
    {
        hod_params.abias = -0.2;
        hod_params.transfP1 = 0.0;
    }
    
    Globals globals (inbase, /*z=*/0.5, /*seed=*/137);

    std::vector<float> xgal;
    std::vector<int64_t> halo_id;

    status = get_galaxies<Cat::Rockstar, secondary, VelMode::None>
        (globals, hod_params, xgal, halo_id, nullptr);
    if (status) return 1;

    std::printf("%ld halos, %ld galaxies (%ld centrals, %ld satellites)\n",
                globals.Nhalos, globals.Ngals, globals.Ncen, globals.Nsat);

    char buffer[512];
    std::sprintf(buffer, "%s_x.bin", outbase);
    save_vec(buffer, xgal);

//    std::sprintf(buffer, "%s_v.bin", outbase);
//    save_vec(buffer, vgal);

    std::sprintf(buffer, "%s_id.bin", outbase);
    save_vec(buffer, halo_id);

    std::vector<float> k, Pk;
    std::vector<int64_t> Nk;

    status = get_power<mas>(globals, xgal, Nmesh, k, Pk, Nk);
    if (status) return 1;

    std::sprintf(buffer, "%s_k.bin", outbase);
    save_vec(buffer, k);
    std::sprintf(buffer, "%s_pk.bin", outbase);
    save_vec(buffer, Pk);
    std::sprintf(buffer, "%s_Nmodes.bin", outbase);
    save_vec(buffer, Nk);

    return 0;
}
