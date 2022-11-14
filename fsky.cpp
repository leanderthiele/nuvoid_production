#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <gsl/gsl_rng.h>

namespace cmangle {
extern "C"
{
    #include "pymangle/mangle.h"
}
}

// where to find masks
const char boss_dir[] = "/tigress/lthiele/boss_dr12";

// survey masks
const char ang_mask_fname[] = "mask_DR12v5_CMASS_North.ply";

const int Nveto = 6;
// we order them by size so the cheap ones go first
const char *veto_fnames[Nveto] =
    {
      "bright_object_mask_rykoff_pix.ply", 
      "centerpost_mask_dr12.ply", 
      "collision_priority_mask_dr12.ply",
      "badfield_mask_postprocess_pixs8.ply", 
      "allsky_bright_star_mask_pix.ply",
      "badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply",
    };

int main(int argc, char **argv)
{
    size_t N = std::atoi(argv[1]);

    cmangle::MangleMask *ang_mask;
    cmangle::MangleMask *veto_masks[Nveto]; // maybe unused

    char mask_fname[512];
    std::sprintf(mask_fname, "%s/%s", boss_dir, ang_mask_fname);
    ang_mask = cmangle::mangle_new();
    cmangle::mangle_read(ang_mask, mask_fname);
    cmangle::set_pixel_map(ang_mask);

    for (int ii=0; ii<Nveto; ++ii)
    {
        std::sprintf(mask_fname, "%s/%s", boss_dir, veto_fnames[ii]);
        veto_masks[ii] = cmangle::mangle_new();
        cmangle::mangle_read(veto_masks[ii], mask_fname);
        cmangle::set_pixel_map(veto_masks[ii]);
    }

    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

    size_t in_footprint = 0;
    for (size_t ii=0; ii<N; ++ii)
    {
        if ((ii+1)%10000==0) std::printf("%lu / %lu\n", ii+1, N);

        double u = gsl_rng_uniform(rng);
        double v = gsl_rng_uniform(rng);

        double phi = 2.0 * M_PI * u;
        double theta = std::acos(2.0 * v - 1.0);

        cmangle::Point pt;
        cmangle::point_set_from_thetaphi(&pt, theta, phi);
        int64_t poly_id; long double weight;
        cmangle::mangle_polyid_and_weight_pix(ang_mask, &pt, &poly_id, &weight);
        if (weight==0.0L) continue;

        bool vetoed = false;
        for (int jj=0; jj<Nveto && !vetoed; ++jj)
        {
            cmangle::mangle_polyid_and_weight_pix(veto_masks[jj], &pt, &poly_id, &weight);
            if (weight!=0.0L) vetoed = true;
        }
        if (vetoed) continue;

        ++in_footprint;
    }

    std::printf("fsky = %.8f\n", (double)in_footprint/(double)N);

    return 0;
}
