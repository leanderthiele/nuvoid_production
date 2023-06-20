#include <cstddef>
#include <vector>
#include <map>
#include <string>
#include <limits>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
using namespace pybind11::literals;
namespace pyb = pybind11;

#include "have_if.h"
#include "enums.h"
#include "globals.h"
#include "hod_params.h"
#include "halo.h"
#include "read_cat.h"
#include "galaxies.h"
#include "power.h"
#include "err.h"
#include "timing.h"

#include "enum_dispatch.h"

// the python return object
struct RetVal
{// {{{
    bool all_cosmo_equal;

    // these are only used if the cosmologies are identical (for convenience)
    uint64_t cosmo_hash_high_word_single, cosmo_hash_low_word_single;
    std::map<std::string, float> cosmo_single;

    // otherwise, we'll use these
    pyb::array_t<uint64_t> cosmo_hash_high_word, cosmo_hash_low_word;
    std::map<std::string, pyb::array_t<float>> cosmo;

    std::map<std::string, pyb::array_t<int64_t>> objects;

    std::map<std::string, float> timing;

    pyb::array_t<float> k, Pk;
    pyb::array_t<int64_t> Nk;
};// }}}

// small helper functions for parallel versions

template<typename T>
static auto assemble_vec (size_t N, const Globals *g, size_t offset)
{// {{{
    std::vector<T> out;
    for (size_t ii=0; ii<N; ++ii)
        out.push_back(*(T *)((char *)(g+ii)+offset));
    return out;
}// }}}

template<typename T=float>
static float avg (size_t N, const Globals *g, size_t offset)
{// {{{
    T out = 0;
    for (size_t ii=0; ii<N; ++ii)
        out += *(T *)((char *)(g+ii)+offset);
    return out / N;
}// }}}

static void finish_output (size_t Nruns, const Globals *globals, RetVal &out)
{// {{{
    out.all_cosmo_equal = true;
    for (size_t ii=1; ii<Nruns; ++ii)
        if (globals[ii].cosmo_hash_low_word != globals[0].cosmo_hash_low_word
            || globals[ii].cosmo_hash_high_word != globals[0].cosmo_hash_high_word)
        { out.all_cosmo_equal = false; break; }

    if (out.all_cosmo_equal)
    {
        out.cosmo_hash_high_word_single = globals[0].cosmo_hash_high_word;
        out.cosmo_hash_low_word_single = globals[0].cosmo_hash_low_word;

        out.cosmo_single["Omega_m"] = globals[0].O_m;
        out.cosmo_single["Omega_b"] = globals[0].O_b;
        out.cosmo_single["Omega_nu"] = globals[0].O_nu;
        out.cosmo_single["Omega_cdm"] = globals[0].O_cdm;
        out.cosmo_single["h"] = globals[0].h;
        out.cosmo_single["n_s"] = globals[0].n_s;
        out.cosmo_single["sigma_8"] = globals[0].sigma_8;
        out.cosmo_single["A_s"] = globals[0].A_s;
    }
    else
    {
        out.cosmo_hash_high_word = pyb::cast(assemble_vec<uint64_t>
            (Nruns, globals, offsetof(Globals, cosmo_hash_high_word)));
        out.cosmo_hash_low_word = pyb::cast(assemble_vec<uint64_t>
            (Nruns, globals, offsetof(Globals, cosmo_hash_low_word)));
        out.cosmo["Omega_m"] = pyb::cast(assemble_vec<float>
            (Nruns, globals, offsetof(Globals, O_m)));
        out.cosmo["Omega_b"] = pyb::cast(assemble_vec<float>
            (Nruns, globals, offsetof(Globals, O_b)));
        out.cosmo["Omega_nu"] = pyb::cast(assemble_vec<float>
            (Nruns, globals, offsetof(Globals, O_nu)));
        out.cosmo["Omega_cdm"] = pyb::cast(assemble_vec<float>
            (Nruns, globals, offsetof(Globals, O_cdm)));
        out.cosmo["h"] = pyb::cast(assemble_vec<float>
            (Nruns, globals, offsetof(Globals, h)));
        out.cosmo["n_s"] = pyb::cast(assemble_vec<float>
            (Nruns, globals, offsetof(Globals, n_s)));
        out.cosmo["sigma_8"] = pyb::cast(assemble_vec<float>
            (Nruns, globals, offsetof(Globals, sigma_8)));
        out.cosmo["A_s"] = pyb::cast(assemble_vec<float>
            (Nruns, globals, offsetof(Globals, A_s)));
    }

    out.objects["Nhalos"] = pyb::cast(assemble_vec<int64_t>
            (Nruns, globals, offsetof(Globals, Nhalos)));
    out.objects["Ngals"] = pyb::cast(assemble_vec<int64_t>
            (Nruns, globals, offsetof(Globals, Ngals)));
    out.objects["Ncen"] = pyb::cast(assemble_vec<int64_t>
            (Nruns, globals, offsetof(Globals, Ncen)));
    out.objects["Nsat"] = pyb::cast(assemble_vec<int64_t>
            (Nruns, globals, offsetof(Globals, Nsat)));

    out.timing["get_galaxies"] = avg(Nruns, globals, offsetof(Globals, time_get_galaxies));
    out.timing["read_cat"] = avg(Nruns, globals, offsetof(Globals, time_read_cat));
    out.timing["populate"] = avg(Nruns, globals, offsetof(Globals, time_populate));
    out.timing["assign_types"] = avg(Nruns, globals, offsetof(Globals, time_assign_types));
    out.timing["draw_gals"] = avg(Nruns, globals, offsetof(Globals, time_draw_gals));
    out.timing["power"] = avg(Nruns, globals, offsetof(Globals, time_power));
    out.timing["mesh"] = avg(Nruns, globals, offsetof(Globals, time_mesh));
    out.timing["mark"] = avg(Nruns, globals, offsetof(Globals, time_mark));
    out.timing["pk"] = avg(Nruns, globals, offsetof(Globals, time_pk));
    out.timing["fft"] = avg(Nruns, globals, offsetof(Globals, time_fft));
    out.timing["bin"] = avg(Nruns, globals, offsetof(Globals, time_bin));
}// }}}

template<RSD rsd, bool binary, bool vgal_separate>
static int write_galaxies (const std::string &fname, const Globals &globals,
                           const std::vector<float> &xgal,
                           HAVE_IF(rsd != RSD::None || vgal_separate, const std::vector<float> &) vgal,
                           HAVE_IF(rsd != RSD::None || vgal_separate, const std::vector<float> &) vhlo)
// if binary, writes in the binary format read by zobov (posread in readfiles.c)
// else, writes in the multidark format read by VIDE
{
    [[maybe_unused]] float rsd_factor;
    if constexpr (rsd != RSD::None || vgal_separate)
        rsd_factor = (1.0F+globals.z)
                     / (100.0F * std::sqrt(globals.O_m*std::pow(1.0F+globals.z, 3) + (1.0F-globals.O_m)));

    [[maybe_unused]] std::vector<float> xcomponents[3];

    if constexpr (!vgal_separate)
        for (int64_t ii=0; ii<globals.Ngals; ++ii)
            for (int jj=0; jj<3; ++jj)
            {
                float this_x = xgal[3*ii+jj];
                if constexpr (rsd != RSD::None)
                    if (jj==(int)rsd)
                        this_x += rsd_factor * vgal[3*ii+jj];

                // make periodic
                float r = std::fmod(this_x, globals.BoxSize);
                this_x = (r<0.0F) ? r+globals.BoxSize : r;

                xcomponents[jj].push_back(this_x);
            }

    auto fp = std::fopen(fname.c_str(), "w");
    CHECK(!fp, return 1);

    if constexpr (binary)
    {
        if constexpr (vgal_separate)
        {
            std::fwrite(xgal.data(), sizeof(float), 3*globals.Ngals, fp);
            std::fwrite(vgal.data(), sizeof(float), 3*globals.Ngals, fp);
            std::fwrite(vhlo.data(), sizeof(float), 3*globals.Ngals, fp);
        }
        else
            for (int ii=0; ii<3; ++ii)
                std::fwrite(xcomponents[ii].data(), sizeof(float), globals.Ngals, fp);

        /* The following is the zobov format which I don't think we want anymore

        const int np = globals.Ngals;

        for (int ii=0; ii<3; ++ii) std::fwrite(&np, sizeof(int), 1, fp);

        for (int ii=0; ii<3; ++ii)
        {
            std::fwrite(&np, sizeof(int), 1, fp); // should be ignored
            std::fwrite(xcomponents[ii].data(), sizeof(float), globals.Ngals, fp);
            std::fwrite(&np, sizeof(int), 1, fp); // should be ignored
        }
        */
    }
    else // txt format, doesn't work with vhlo
    {
        // the header in multidark convention
        std::fprintf(fp, "%.1f\n%.4f\n%.4f\n%.4f\n%ld\n",
                         globals.BoxSize, globals.O_m, globals.h,
                         globals.z, globals.Ngals);

        // the galaxy positions
        for (int64_t ii=0; ii<globals.Ngals; ++ii)
            if constexpr (vgal_separate)
                std::fprintf(fp, "%ld %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                 ii,
                                 xgal[3*ii+0], xgal[3*ii+1], xgal[3*ii+2],
                                 vgal[3*ii+0], vgal[3*ii+1], vgal[3*ii+2],
                                 0.0);
            else
                std::fprintf(fp, "%ld %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                 ii, xcomponents[0][ii], xcomponents[1][ii], xcomponents[2][ii],
                                 0.0, 0.0, 0.0, 0.0);
        
        // the final line
        std::fprintf(fp, "-99 -99 -99 -99 -99 -99 -99 -99 -99\n");
    }

    std::fclose(fp);

    return 0;
}


template<int cat_, int secondary_, int have_vbias_, int have_zdep_>
struct py_get_galaxies_templ
// quick convenience function for our concrete application
{
static void apply
    (const std::string base, const std::vector<double> times, const std::string galaxies_bin_base,
     float hod_log_Mmin, float hod_sigma_logM,
     float hod_log_M0, float hod_log_M1,
     float hod_alpha,
     [[maybe_unused]] float hod_transfP1,
     [[maybe_unused]] float hod_abias,
     [[maybe_unused]] float hod_transf_eta_cen,
     [[maybe_unused]] float hod_transf_eta_sat,
     [[maybe_unused]] float hod_mu_Mmin,
     [[maybe_unused]] float hod_mu_M1,
     int64_t seed,
     HaloDef mdef)
{// {{{
    constexpr auto cat = static_cast<Cat>(cat_);
    constexpr auto secondary = static_cast<Sec>(secondary_);
    constexpr auto have_vbias = static_cast<bool>(have_vbias_);
    constexpr auto have_zdep = static_cast<bool>(have_zdep_);

    HODParams<secondary != Sec::None, have_vbias, have_zdep> hod_params;
    hod_params.log_Mmin = hod_log_Mmin; hod_params.sigma_logM = hod_sigma_logM;
    hod_params.log_M0 = hod_log_M0; hod_params.log_M1 = hod_log_M1;
    hod_params.alpha= hod_alpha;
    if constexpr (secondary != Sec::None)
    {
        hod_params.transfP1 = hod_transfP1;
        hod_params.abias = hod_abias;
    }
    if constexpr (have_vbias)
    {
        hod_params.transf_eta_cen = hod_transf_eta_cen;
        hod_params.transf_eta_sat = hod_transf_eta_sat;
    }
    if constexpr (have_zdep)
    {
        hod_params.mu_Mmin = hod_mu_Mmin;
        hod_params.mu_M1 = hod_mu_M1;
    }

    const size_t Nruns = times.size();

    int status[Nruns];
    Globals globals[Nruns];

    #pragma omp parallel for
    for (size_t ii=0; ii<Nruns; ++ii)
    {
        globals[ii] = Globals(base.c_str(), (float)(1.0/times[ii]-1.0), seed, mdef);

        std::vector<float> xgal, vgal, vhlo;
        status[ii] = get_galaxies<cat, secondary, (have_vbias) ? VelMode::Biased : VelMode::Unbiased>
            (globals[ii], hod_params, xgal, vgal, vhlo);
        if (status[ii]) continue;

        char fname[512];
        std::sprintf(fname, "%s_%.4f.bin", galaxies_bin_base.c_str(), times[ii]);
        status[ii] = write_galaxies<RSD::None, /*binary=*/true, /*vgal_separate*/true>
            (fname, globals[ii], xgal, vgal, vhlo);
    }

    for (size_t ii=0; ii<Nruns; ++ii) CHECK(status[ii], throw std::runtime_error("failed!"));
}// }}}
     
static constexpr bool allowed ()
{// {{{
    constexpr auto cat = static_cast<Cat>(cat_);
    constexpr auto secondary = static_cast<Sec>(secondary_);

    if constexpr ((cat==Cat::FOF || cat==Cat::RFOF) && (secondary != Sec::None))
        return false;
    
    return true;
}// }}}
};

void py_get_galaxies
    (const std::string base, const std::vector<double> times, const std::string galaxies_bin_base,
     Cat cat=Cat::Rockstar, Sec secondary=Sec::None,
     float hod_log_Mmin=13.03, float hod_sigma_logM=0.38,
     float hod_log_M0=13.27, float hod_log_M1=14.08,
     float hod_alpha=0.76, float hod_transfP1=0.0, float hod_abias=0.0,
     bool have_vbias=false,
     // ignore if not have_vbias, fiducial values (corresponding to no bias) are 0
     float hod_transf_eta_cen=0.0, float hod_transf_eta_sat=0.0,
     bool have_zdep=false,
     float hod_mu_Mmin=0.0, float hod_mu_M1=0.0,
     int64_t seed=std::numeric_limits<int64_t>::max(),
     HaloDef mdef=HaloDef::v)
{
    const auto dispatcher = Dispatcher<py_get_galaxies_templ, Cat, Sec, bool, bool>();
    
    return dispatcher(cat, secondary, have_vbias, have_zdep)
        (base, times, galaxies_bin_base,
         hod_log_Mmin, hod_sigma_logM,
         hod_log_M0, hod_log_M1,
         hod_alpha, hod_transfP1, hod_abias,
         hod_transf_eta_cen, hod_transf_eta_sat,
         hod_mu_Mmin, hod_mu_M1,
         seed, mdef);
}



template<int cat_, int secondary_, int mas_, int rsd_, int have_vbias_, int have_mark_, int vgal_separate_, int have_zdep_>
struct py_get_power_templ
{
static RetVal apply
    (const std::vector<std::string> base, float z,
     float hod_log_Mmin, float hod_sigma_logM,
     float hod_log_M0, float hod_log_M1,
     float hod_alpha,
     [[maybe_unused]] float hod_transfP1,
     [[maybe_unused]] float hod_abias,
     [[maybe_unused]] float hod_transf_eta_cen,
     [[maybe_unused]] float hod_transf_eta_sat,
     [[maybe_unused]] float hod_mu_Mmin,
     [[maybe_unused]] float hod_mu_M1,
     [[maybe_unused]] float mark_p,
     [[maybe_unused]] float mark_delta_s,
     [[maybe_unused]] float mark_R,
     int64_t Nmesh, float kmax, int64_t seed,
     HaloDef mdef,
     const std::string galaxies_bin_base,
     const std::string galaxies_txt_base)
{// {{{
    constexpr auto cat = static_cast<Cat>(cat_);
    constexpr auto secondary = static_cast<Sec>(secondary_);
    constexpr auto mas = static_cast<MAS>(mas_);
    constexpr auto rsd = static_cast<RSD>(rsd_);
    constexpr auto have_vbias = static_cast<bool>(have_vbias_);
    constexpr auto have_mark = static_cast<bool>(have_mark_);
    constexpr auto vgal_separate = static_cast<bool>(vgal_separate_);
    constexpr auto have_zdep = static_cast<bool>(have_zdep_);

    auto start = start_time();

    // the rsd != None in the following should not be necessary as the allowed()
    // guard below takes care of this, but somehow my clang complains if I don't
    // include it
    HODParams<secondary != Sec::None, have_vbias && (rsd!=RSD::None || vgal_separate), have_zdep> hod_params;
    hod_params.log_Mmin = hod_log_Mmin; hod_params.sigma_logM = hod_sigma_logM;
    hod_params.log_M0 = hod_log_M0; hod_params.log_M1 = hod_log_M1;
    hod_params.alpha= hod_alpha;
    if constexpr (secondary != Sec::None)
    {
        hod_params.transfP1 = hod_transfP1;
        hod_params.abias = hod_abias;
    }
    if constexpr (have_vbias)
    {
        hod_params.transf_eta_cen = hod_transf_eta_cen;
        hod_params.transf_eta_sat = hod_transf_eta_sat;
    }
    if constexpr (have_zdep)
    {
        hod_params.mu_Mmin = hod_mu_Mmin;
        hod_params.mu_M1 = hod_mu_M1;
    }

    const size_t Nruns = base.size();

    int status[Nruns];
    Globals globals[Nruns];

    // for the moment, we assume that these are identical
    std::vector<float> k;
    std::vector<int64_t> Nk;

    std::vector<float> Pk[Nruns];

    #pragma omp parallel for
    for (size_t ii=0; ii<Nruns; ++ii)
    {
        globals[ii] = Globals(base[ii].c_str(), z, seed, mdef);

        std::vector<float> xgal;
        HAVE_IF(rsd != RSD::None || vgal_separate, std::vector<float>) vgal;
        HAVE_IF(rsd != RSD::None || vgal_separate, std::vector<float>) vhlo;
        status[ii] = get_galaxies<cat, secondary,
                                  (rsd==RSD::None && !vgal_separate) ? VelMode::None
                                  : (have_vbias) ? VelMode::Biased : VelMode::Unbiased,
                                  have_zdep>
            (globals[ii], hod_params, xgal, vgal, vhlo);
        if (status[ii]) continue;

        if (galaxies_bin_base.size()) // empty string corresponds to no write
        {
            char fname[512];
            std::sprintf(fname, "%s_%.4f.bin", galaxies_bin_base.c_str(), 1.0/(1.0+z));
            status[ii] = write_galaxies<rsd, /*binary=*/true, vgal_separate>(fname, globals[ii], xgal, vgal, vhlo);
        }
        
        if (galaxies_txt_base.size())
        {
            char fname[512];
            std::sprintf(fname, "%s_%.4f.txt", galaxies_txt_base.c_str(), 1.0/(1.0+z));
            status[ii] = write_galaxies<rsd, /*binary=*/false, vgal_separate>(fname, globals[ii], xgal, vgal, vhlo);
        }

        std::vector<float> this_k; std::vector<int64_t> this_Nk;
        status[ii] = get_power<mas, rsd, have_mark>
            (globals[ii], xgal, vgal, mark_p, mark_delta_s, mark_R,
             Nmesh, kmax, (ii) ? this_k : k, Pk[ii], (ii) ? this_Nk : Nk);
        if (status[ii]) continue;
    }

    for (size_t ii=0; ii<Nruns; ++ii) CHECK(status[ii], throw std::runtime_error("failed!"));

    RetVal out;

    out.k = pyb::cast(k);
    out.Nk = pyb::cast(Nk);

    auto Nbins = Pk[0].size();
    Pk[0].reserve(Nruns * Nbins);
    for (size_t ii=1; ii<Nruns; ++ii)
    {
        CHECK(Pk[ii].size() != Nbins, throw std::runtime_error("Pk have different sizes"));
        Pk[0].insert(Pk[0].end(), Pk[ii].begin(), Pk[ii].end());
    }

    out.Pk = pyb::cast(Pk[0]);

    if constexpr (rsd == RSD::None)
        out.Pk = out.Pk.reshape({Nruns, k.size()});
    else
        out.Pk = out.Pk.reshape({Nruns, 3UL, k.size()});

    finish_output(Nruns, globals, out);
    out.timing["total"] = get_time(start);

    return out;
}// }}}

static constexpr bool allowed ()
{// {{{
    constexpr auto cat = static_cast<Cat>(cat_);
    constexpr auto secondary = static_cast<Sec>(secondary_);
    [[maybe_unused]] constexpr auto mas = static_cast<MAS>(mas_);
    constexpr auto rsd = static_cast<RSD>(rsd_);
    constexpr auto have_vbias = static_cast<bool>(have_vbias_);
    constexpr auto vgal_separate = static_cast<bool>(vgal_separate_);

    if constexpr ((cat==Cat::FOF || cat==Cat::RFOF) && (secondary != Sec::None))
        return false;
    if constexpr ((rsd == RSD::None) && have_vbias && !vgal_separate)
        return false;

    return true;
}// }}}
};

RetVal py_get_power
    (const std::vector<std::string> base, float z,
     RSD rsd=RSD::None, Cat cat=Cat::Rockstar, Sec secondary=Sec::None,
     float hod_log_Mmin=13.03, float hod_sigma_logM=0.38,
     float hod_log_M0=13.27, float hod_log_M1=14.08,
     float hod_alpha=0.76, float hod_transfP1=0.0, float hod_abias=0.0,
     bool have_vbias=false,
     // ignore if not have_vbias, fiducial values (corresponding to no bias) are 0
     float hod_transf_eta_cen=0.0, float hod_transf_eta_sat=0.0,
     bool have_zdep=false,
     float hod_mu_Mmin=0.0, float hod_mu_M1=0.0,
     bool have_mark=false,
     // ignore if not have_mark, fiducial values from 2001.11024
     float mark_p=2.0, float mark_delta_s=0.25, float mark_R=10.0,
     int64_t Nmesh=512, float kmax=0.6, MAS mas=MAS::PCS,
     int64_t seed=std::numeric_limits<int64_t>::max(),
     HaloDef mdef=HaloDef::v,
     const std::string galaxies_bin_base="",
     const std::string galaxies_txt_base="",
     bool vgal_separate=false)
{// {{{
    const auto dispatcher = Dispatcher<py_get_power_templ, Cat, Sec, MAS, RSD, bool, bool, bool, bool>();

    return dispatcher(cat, secondary, mas, rsd, have_vbias, have_mark, vgal_separate, have_zdep)
        (base, z,
         hod_log_Mmin, hod_sigma_logM,
         hod_log_M0, hod_log_M1,
         hod_alpha, hod_transfP1, hod_abias,
         hod_transf_eta_cen, hod_transf_eta_sat,
         hod_mu_Mmin, hod_mu_M1,
         mark_p, mark_delta_s, mark_R,
         Nmesh, kmax, seed, mdef,
         galaxies_bin_base, galaxies_txt_base);
}// }}}

template<int cat_, int mas_>
struct py_get_halo_power_templ
{
static RetVal apply
    (const std::vector<std::string> base, float z,
     float Mmin,
     int64_t Nmesh)
{// {{{
    constexpr auto cat = static_cast<Cat>(cat_);
    constexpr auto mas = static_cast<MAS>(mas_);

    auto start = start_time();

    const size_t Nruns = base.size();

    int status[Nruns];
    Globals globals[Nruns];

    std::vector<float> k;
    std::vector<int64_t> Nk;

    std::vector<float> Pk[Nruns];

    #pragma omp parallel for
    for (size_t ii=0; ii<Nruns; ++ii)
    {
        globals[ii] = Globals(base[ii].c_str(), z, 0/*doesn't matter*/, HaloDef::v/*doesn't matter*/);
        std::vector<Halo<false>> halos;
        status[ii] = read_cat<cat, Sec::None, false> (globals[ii], halos);
        if (status[ii]) continue;

        std::vector<float> xhalo;
        for (const auto &h: halos)
            if (h.M > Mmin)
                for (int jj=0; jj<3; ++jj)
                    xhalo.push_back(h.pos[jj]);

        // this is quite hacky...
        globals[ii].Ngals = xhalo.size() / 3;

        std::vector<float> this_k; std::vector<int64_t> this_Nk;
        status[ii] = get_power<mas, RSD::None, false>
            (globals[ii], xhalo, /*velocities*/nullptr, /*3 x mark*/nullptr, nullptr, nullptr,
             Nmesh, /*kmax*/1e2, (ii) ? this_k : k, Pk[ii], (ii) ? this_Nk : Nk);
        if (status[ii]) continue;
    }

    for (size_t ii=0; ii<Nruns; ++ii) CHECK(status[ii], throw std::runtime_error("failed!"));

    RetVal out;

    out.k = pyb::cast(k);
    out.Nk = pyb::cast(Nk);

    Pk[0].reserve(Nruns * Pk[0].size());
    for (size_t ii=1; ii<Nruns; ++ii)
    {
        CHECK(Pk[ii].size() != k.size(), throw std::runtime_error("Pk have different sizes"));
        Pk[0].insert(Pk[0].end(), Pk[ii].begin(), Pk[ii].end());
    }

    out.Pk = pyb::cast(Pk[0]);
    out.Pk = out.Pk.reshape({Nruns, k.size()});

    finish_output(Nruns, globals, out);
    out.timing["total"] = get_time(start);

    return out;
}// }}}

static constexpr bool allowed () { return true; }
};

RetVal py_get_halo_power
    (const std::vector<std::string> base, float z, float Mmin,
     Cat cat=Cat::Rockstar, MAS mas=MAS::PCS,
     int64_t Nmesh=512)
{
    const auto dispatcher = Dispatcher<py_get_halo_power_templ, Cat, MAS>();

    return dispatcher(cat, mas) (base, z, Mmin, Nmesh);
}

/* PYBIND11 DECLARATIONS */

PYBIND11_MODULE(pyglx, m)
{
    // some enumerations

    pyb::module_ mHaloDef = m.def_submodule("HaloDef");
    pyb::enum_<HaloDef>(mHaloDef, "HaloDef")
        .value("c", HaloDef::c)
        .value("v",  HaloDef::v)
        .value("m", HaloDef::m)
        .export_values();
    
    pyb::module_ mCatalogType = m.def_submodule("CatalogType");
    pyb::enum_<Cat>(mCatalogType, "CatalogType")
        .value("rockstar", Cat::Rockstar)
        .value("fof", Cat::FOF)
        .value("rfof", Cat::RFOF)
        .export_values();

    pyb::module_ mSecondary = m.def_submodule("Secondary");
    pyb::enum_<Sec>(mSecondary, "Secondary")
        .value("none", Sec::None) // otherwise python doesn't like us
        .value("conc", Sec::Conc)
        .value("kinpot", Sec::TU)
        .export_values();
    
    pyb::module_ mMAS = m.def_submodule("MAS");
    pyb::enum_<MAS>(mMAS, "MAS")
        .value("ngp", MAS::NGP)
        .value("cic", MAS::CIC)
        .value("tsc", MAS::TSC)
        .value("pcs", MAS::PCS)
        .export_values();

    pyb::module_ mRSD = m.def_submodule("RSD");
    pyb::enum_<RSD>(mRSD, "RSD")
        .value("x", RSD::x)
        .value("y", RSD::y)
        .value("z", RSD::z)
        .value("none", RSD::None)
        .export_values();

    // return structs

    pyb::class_<RetVal> (m, "RetVal")
        .def_readonly("all_cosmo_equal", &RetVal::all_cosmo_equal)
        .def_readonly("cosmo_hash_high_word_single", &RetVal::cosmo_hash_high_word_single)
        .def_readonly("cosmo_hash_low_word_single", &RetVal::cosmo_hash_low_word_single)
        .def_readonly("cosmo_single", &RetVal::cosmo_single)
        .def_readonly("cosmo_hash_high_word", &RetVal::cosmo_hash_high_word)
        .def_readonly("cosmo_hash_low_word", &RetVal::cosmo_hash_low_word)
        .def_readonly("cosmo", &RetVal::cosmo)
        .def_readonly("objects", &RetVal::objects)
        .def_readonly("k", &RetVal::k)
        .def_readonly("Pk", &RetVal::Pk)
        .def_readonly("Nk", &RetVal::Nk)
        .def_readonly("timing", &RetVal::timing);

    // functions

    m.def("get_galaxies", &py_get_galaxies,
          "base"_a, "times"_a, "galaxies_bin_base"_a,
          pyb::pos_only(),
          pyb::kw_only(),
          "cat"_a=Cat::Rockstar, "secondary"_a=Sec::None,
          "hod_log_Mmin"_a=13.03, "hod_sigma_logM"_a=0.38,
          "hod_log_M0"_a=13.27, "hod_log_M1"_a=14.08,
          "hod_alpha"_a=0.76, "hod_transfP1"_a=0.0, "hod_abias"_a=0.0,
          "have_vbias"_a=false,
          "hod_transf_eta_cen"_a=0.0, "hod_transf_eta_sat"_a=0.0,
          "have_zdep"_a=false,
          "hod_mu_Mmin"_a=0.0, "hod_mu_M1"_a=0.0,
          "seed"_a=std::numeric_limits<int64_t>::max(),
          "mdef"_a=HaloDef::v);

    m.def("get_power", &py_get_power,
          "base"_a, "z"_a,
          pyb::pos_only(),
          pyb::kw_only(),
          "rsd"_a=RSD::None, "cat"_a=Cat::Rockstar, "secondary"_a=Sec::None,
          "hod_log_Mmin"_a=13.03, "hod_sigma_logM"_a=0.38,
          "hod_log_M0"_a=13.27, "hod_log_M1"_a=14.08,
          "hod_alpha"_a=0.76, "hod_transfP1"_a=0.0, "hod_abias"_a=0.0,
          "have_vbias"_a=false,
          "hod_transf_eta_cen"_a=0.0, "hod_transf_eta_sat"_a=0.0,
          "have_zdep"_a=false,
          "hod_mu_Mmin"_a=0.0, "hod_mu_M1"_a=0.0,
          "have_mark"_a=false,
          "mark_p"_a=2.0, "mark_delta_s"_a=0.25, "mark_R"_a=10.0,
          "Nmesh"_a=512, "kmax"_a=0.6,"mas"_a=MAS::PCS,
          "seed"_a=std::numeric_limits<int64_t>::max(),
          "mdef"_a=HaloDef::v,
          "galaxies_bin_base"_a="",
          "galaxies_txt_base"_a="",
          "vgal_separate"_a=false);

    m.def("get_halo_power", &py_get_halo_power,
          "base"_a, "z"_a,
          pyb::pos_only(),
          "Mmin"_a,
          pyb::kw_only(),
          "cat"_a=Cat::Rockstar, "mas"_a=MAS::PCS,
          "Nmesh"_a=512);
}
