/* Main HOD functionality: populate a halo catalog with galaxies */
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>
#include <cstdio>

#include "have_if.h"
#include "halo.h"
#include "globals.h"
#include "hod_params.h"
#include "halo_model.h"
#include "nfw.h"
#include "err.h"
#include "timing.h"

#include "populate.h"

// get from our transformed parameterization to the physical parameters
[[maybe_unused]] static inline float getP1 (float transfP1)
{ return 0.5*(std::tanh(transfP1)+1.0F); }

[[maybe_unused]] static inline float get_eta_sat (float transf_eta_sat)
{ return std::exp(transf_eta_sat); }

// this is a bit awkward because the fiducial point (0) is directly at the
// edge of the allowed interval. However, this transformation should be fine
// since transf_eta_cen=0 corresponds to eta_cen=0 for all purposes.
[[maybe_unused]] static inline float get_eta_cen (float transf_eta_cen)
{ return std::exp(-(10.0F-transf_eta_cen)); }

template<bool have_abias, VelMode vmode, bool have_zdep>
int draw_gals (Globals &globals, const HODParams<have_abias, vmode==VelMode::Biased, have_zdep> &hod_params,
               const std::vector<Halo<have_abias>> &halos,
               // output, take these as vectors since it is useful to be able to resize
               std::vector<float> &xgal,
               HAVE_IF(vmode != VelMode::None, std::vector<float> &) vgal,
               HAVE_IF(vmode != VelMode::None, std::vector<float> &) vhlo)
{
    // some guess of the output size, not very important
    const size_t estimated_Ngal = halos.size() / 10;
    xgal.reserve(3*estimated_Ngal);
    if constexpr (vmode != VelMode::None)
    {
        vgal.reserve(3*estimated_Ngal); 
        vhlo.reserve(3*estimated_Ngal);
    }

    // distributions we need to draw from
    std::default_random_engine rng (globals.seed);
    auto U01 = std::bind(std::uniform_real_distribution<float>(0.0, 1.0), rng);
    auto dist_binary = [&U01](float p) -> bool
        { return U01() < p; };
    auto dist_poisson = [&rng](float mu) -> int
        { return std::poisson_distribution<int>(mu)(rng); };
    auto dist_normal = std::bind(std::normal_distribution<float>(0.0, 1.0), rng);
    auto dist_onsphere = [&U01](const float r0[3], float R, float out[3]) -> void
        // generates a random point on the sphere of radius R around r0
        {
            auto phi = 2.0F * M_PIf32 * U01(),
                 costheta = 2.0F * U01() - 1.0F;
            auto sintheta = std::sqrt(1.0F - costheta*costheta);
            out[0] = r0[0] + R * sintheta * std::cos(phi);
            out[1] = r0[1] + R * sintheta * std::sin(phi);
            out[2] = r0[2] + R * costheta;
        };
    auto dist_Rnfw = [&U01](float c) -> float
        // generates a random galaxy position (for satellites) for a halo with concentration c,
        // Output in units of the boundary radius. We make sure it is in [0, 1]
        // This uses the analytic solution from [1805.09550]
        {
            return std::clamp(NFW::q(c, U01()), 0.0, 1.0);
        };
    [[maybe_unused]] auto dist_Vnfw = [&dist_normal](float c, float r, float out[3]) -> void
        // generates a random galaxy velocity (for satellites) for a halo with concentration c,
        // the galaxy being at radius r (in units of the boundary radius) from the halo center.
        // Output in units of the virial velocity
        {
            float sigma = c * r * (1.0+c*r) * std::sqrt( NFW::zeta(c*r) / NFW::g(c) );
            for (int ii=0; ii<3; ++ii)
                out[ii] = dist_normal() * sigma;
        };
    
    // concentration model we use, can easily change this
    auto conc = std::bind(Duffy08, std::placeholders::_1, globals.z, globals.mdef);

    // virial velocity calculation, can easily change
    [[maybe_unused]] auto vir_vel = std::bind(VofM, std::placeholders::_1, globals.O_m, globals.z, globals.mdef);

    // halo radius calculation, can easily change
    auto radius = std::bind(RofM, std::placeholders::_1, globals.O_m, globals.z, globals.mdef);

    globals.Ngals = 0; globals.Ncen = 0; globals.Nsat = 0;

    // possibly redshift dependent quantities
    [[maybe_unused]] auto fz = [&globals](float logM, float mu) -> float
        {
            // in principle arbitrary but useful to match the effective redshift
            // so we can translate parameters easily
            static const float zpivot = 0.53;

            return logM + mu * (1.0F/(1.0F+globals.z)  - 1.0F/(1.0F+zpivot));
        };
    float this_log_Mmin, this_log_M1;
    if constexpr (have_zdep)
    {
        this_log_Mmin = fz(hod_params.log_Mmin, hod_params.mu_Mmin);
        this_log_M1 = fz(hod_params.log_M1, hod_params.mu_M1);
    }
    else
    {
        this_log_Mmin = hod_params.log_Mmin;
        this_log_M1 = hod_params.log_M1;
    }

    for (auto h: halos)
    {
        const auto logM = std::log10(h.M);
        
        // Zheng+07 model
        auto Ncen = 0.5F * ( 1.0F + std::erf((logM-this_log_Mmin)/hod_params.sigma_logM) );

        const auto M0 = std::exp(M_LN10*hod_params.log_M0);
        const auto M1 = std::exp(M_LN10*this_log_M1);

        auto Nsat = (h.M > M0) ? Ncen * std::pow(( h.M - M0 ) / M1, hod_params.alpha) : 0.0F;

        // the simple Heaviside assembly bias implementation
        if constexpr (have_abias)
        {
            float P1 = getP1(hod_params.transfP1);

            float scale_cen, scale_sat;
            if (h.type) // type 1
            {
                scale_cen = std::min(1.0F-Ncen, (1.0F-P1)/P1 * Ncen);
                scale_sat = (1.0F-P1)/P1 * Nsat;
            }
            else // type 2
            {
                scale_cen = std::max(-Ncen, (1.0F-P1)/P1 * (Ncen-1.0F));
                scale_sat = -Nsat;
            }

            Ncen += std::fabs(hod_params.abias) * scale_cen;
            Nsat += std::fabs(hod_params.abias) * scale_sat;
        }

        // now draw the galaxy numbers
        bool has_cen = dist_binary(Ncen);
        int Nsat_drawn = (Nsat) ? dist_poisson(Nsat) : 0;

        // for each galaxy, store the underlying halo velocity
        if constexpr (vmode != VelMode::None)
            for (int ii=0; ii<(int)(has_cen) + Nsat_drawn; ++ii)
                for (int jj=0; jj<3; ++jj)
                    vhlo.push_back(h.vel[jj]);

        if (has_cen)
        {
            // we have one more central
            ++globals.Ngals;
            ++globals.Ncen;

            // central sits at the host halo position
            for (int ii=0; ii<3; ++ii)
                xgal.push_back(h.pos[ii]);

            if constexpr (vmode == VelMode::Unbiased)
                for (int ii=0; ii<3; ++ii)
                    vgal.push_back(h.vel[ii]);
            else if constexpr (vmode == VelMode::Biased)
            {
                auto sigmav = vir_vel(h.M) * get_eta_cen(hod_params.transf_eta_cen);
                for (int ii=0; ii<3; ++ii)
                    vgal.push_back(h.vel[ii] + sigmav * dist_normal());
            }
        }

        for (int ii=0; ii<Nsat_drawn; ++ii)
        {
            // one more satellite
            ++globals.Ngals;
            ++globals.Nsat;

            auto Rsat = dist_Rnfw(conc(h.M)); // this is in units of the boundary radius, sure in [0, 1)
            float x[3];
            dist_onsphere(h.pos, Rsat * radius(h.M), x);
            for (int jj=0; jj<3; ++jj)
                xgal.push_back(x[jj]);

            if constexpr (vmode != VelMode::None)
            {
                // random satellite velocity in units of the virial velocity
                float v[3];
                dist_Vnfw(conc(h.M), Rsat, v);

                if constexpr (vmode == VelMode::Biased)
                    for (int jj=0; jj<3; ++jj)
                        v[jj] *= get_eta_sat(hod_params.transf_eta_sat);

                for (int jj=0; jj<3; ++jj)
                    vgal.push_back(h.vel[jj] + vir_vel(h.M) * v[jj]);
            }
        }
    }

    return 0;
}

template<bool have_vbias, bool have_zdep>
int assign_types (const HODParams<true, have_vbias, have_zdep> &hod_params,
                  std::vector<Halo<true>> &halos)
// assigns types to the individual halos, based on a binary partition of the property stored in phalo.
// The percentile P1 is interpreted along the increasing phalo direction.
// If sign(abias)=+, the halos with large phalo are assigned true,
// vice versa if sign(abias)=-
// This routine shuffles the halos!
//
// NOTE this routine takes about a second for a QUIJOTE box
{
    const static int64_t Nbins = 64; // number of mass bins to use, for each we do the partition separately

    float P1 = getP1(hod_params.transfP1);

    // first sort the halos by mass
    std::sort(halos.begin(), halos.end(), [](Halo<true> h1, Halo<true> h2) { return h1.M < h2.M; });

    // number of halos in each bin (last one is maybe slightly different)
    int64_t Nperbin = std::lround((float)halos.size() / (float)Nbins);

    // running counter of the first element in the bin
    auto begin = halos.begin();

    // now do the binary partition for each mass bin
    for (int64_t ii=1; ii<=Nbins; ++ii)
    {
        decltype(begin) end;
        if (ii != Nbins)
            end = begin + Nperbin;
        else
            end = halos.end();
        
        CHECK(end > halos.end(), return 1);

        auto nth = begin + std::lround(P1 * (end-begin));
        std::nth_element(begin, nth, end,
                         [](Halo<true> h1, Halo<true> h2)
                         { return h1.abias_property > h2.abias_property; });

        for (auto h=begin; h<nth; ++h)
            h->type = hod_params.abias < 0.0;
        for (auto h=nth; h<end; ++h)
            h->type = hod_params.abias > 0.0;

        // and increase our pointer
        begin = end;
    }

    return 0;
}

template<bool have_abias, VelMode vmode, bool have_zdep>
int populate (Globals &globals, const HODParams<have_abias, vmode==VelMode::Biased, have_zdep> &hod_params,
              std::vector<Halo<have_abias>> &halos,
              std::vector<float> &xgal,
              HAVE_IF(vmode != VelMode::None, std::vector<float> &) vgal,
              HAVE_IF(vmode != VelMode::None, std::vector<float> &) vhlo)
{
    int status;

    auto start = start_time();

    if constexpr (have_abias)
    {
        auto start1 = start_time();
        status = assign_types<>(hod_params, halos);
        CHECK(status, return 1);
        globals.time_assign_types = get_time(start1);
    }

    auto start2 = start_time();
    status = draw_gals<have_abias, vmode, have_zdep>(globals, hod_params, halos, xgal, vgal, vhlo);
    CHECK(status, return 1);
    globals.time_draw_gals = get_time(start2);

    globals.time_populate = get_time(start);

    return 0;
}

// explicit instantiations
#define INSTANTIATE(have_abias, vmode, have_zdep) \
    template int populate<have_abias, vmode, have_zdep> \
    (Globals &, const HODParams<have_abias, vmode==VelMode::Biased, have_zdep> &,  \
     std::vector<Halo<have_abias>> &, \
     std::vector<float> &, \
     HAVE_IF(vmode != VelMode::None, std::vector<float> &), \
     HAVE_IF(vmode != VelMode::None, std::vector<float> &))

INSTANTIATE(true, VelMode::None, false);
INSTANTIATE(true, VelMode::Unbiased, false);
INSTANTIATE(true, VelMode::Biased, false);
INSTANTIATE(false, VelMode::None, false);
INSTANTIATE(false, VelMode::Unbiased, false);
INSTANTIATE(false, VelMode::Biased, false);
INSTANTIATE(true, VelMode::None, true);
INSTANTIATE(true, VelMode::Unbiased, true);
INSTANTIATE(true, VelMode::Biased, true);
INSTANTIATE(false, VelMode::None, true);
INSTANTIATE(false, VelMode::Unbiased, true);
INSTANTIATE(false, VelMode::Biased, true);
