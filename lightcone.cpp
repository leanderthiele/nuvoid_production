#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <functional>
#include <map>
#include <utility>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_rng.h>

#include "cuboid.h"

namespace cmangle {
extern "C"
{
    #include "pymangle/mangle.h"
}
}

#include "healpix_base.h"

// these are the possible remaps I found
int remaps[][9] =
                  { // 1.4142 1.0000 0.7071
                    { 1, 1, 0,
                      0, 0, 1,
                      1, 0, 0, },
                    // 1.7321 0.8165 0.7071
                    { 1, 1, 1,
                      1, 0, 0,
                      0, 1, 0, },
                  };

// get from the quadrant ra=[-90,90], dec=[0,90] to the NGC footprint
// we only need a rotation around the y-axis I believe
const double alpha = 97.0 * M_PI / 180.0; // rotation around y-axis
const double beta = 6.0; // rotation around z-axis, in degrees

// in units of L1, L2, L3
const double origin[] = { 0.5, -0.058, 0.0 };

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

// how many bins we use for redshift histogramming
// should give redshift slices that are larger than the correlation
// length but still capture any evolution
const int N_zbins = 80;

// used for the initial z downsampling, needs to be adjusted
// In experiments, I found 2.09 %
const double fibcoll_rate = 0.025;

template<bool reverse>
int dbl_cmp (const void *a_, const void *b_)
{
    double a = *(double *)a_;
    double b = *(double *)b_;
    int sgn = (reverse) ? -1 : 1;
    return sgn * ( (a>b) ? +1 : -1 );
}

double integrand (double z, void *p)
{
    static const double H0inv = 2.99792458e3; // Mpc/h
    double Omega_m = *(double *)p;
    return H0inv / std::sqrt( Omega_m*gsl_pow_3(1.0+z)+(1.0-Omega_m) );
}

void comoving (double Omega_m, int N, double *z, double *chi)
// populate chi with chi(z) in Mpc/h
{
    gsl_function F;
    F.function = integrand;
    F.params = &Omega_m;

    static const size_t ws_size = 1024;
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(ws_size);

    double err;
    for (int ii=0; ii<N; ++ii)
        gsl_integration_qag(&F, 0.0, z[ii], 1e-2, 0.0, ws_size, 6, ws, chi+ii, &err);

    gsl_integration_workspace_free(ws);
}

std::vector<double> read_boss_z (const char *boss_dir)
{
    char fname[512];
    std::sprintf(fname, "%s/galaxy_DR12v5_CMASS_North.bin", boss_dir);
    auto fp = std::fopen(fname, "rb");
    // binary file containing ra, dec, z in this order in double
    
    // figure out number of galaxies
    std::fseek(fp, 0, SEEK_END);
    auto nbytes = std::ftell(fp);

    auto Ngal = nbytes/sizeof(double)/3;
    if (!(Ngal*3*sizeof(double)==nbytes)) throw std::runtime_error("read_boss_z faield");

    // skip RA and DEC
    std::fseek(fp, 2*Ngal*sizeof(double), SEEK_SET);

    std::vector<double> out;
    out.resize(Ngal);

    std::fread(out.data(), sizeof(double), Ngal, fp);

    std::fclose(fp);

    return out;
}

void downsample (gsl_histogram *target_hist, double plus_factor,
                 std::vector<double> &ra_vec, std::vector<double> &dec_vec, std::vector<double> &z_vec)
{
    gsl_histogram *sim_z_hist = gsl_histogram_clone(target_hist); // get the same ranges
    gsl_histogram_reset(sim_z_hist);
    for (auto z : z_vec)
        gsl_histogram_increment(sim_z_hist, z);
    double keep_fraction[target_hist->n];
    for (int ii=0; ii<target_hist->n; ++ii)
        // note the GSL bin counts are doubles already
        keep_fraction[ii] = (1.0+plus_factor)
                            * gsl_histogram_get(target_hist, ii) / gsl_histogram_get(sim_z_hist, ii);

    // for debugging
    std::printf("keep_fraction:\n");
    for (int ii=0; ii<target_hist->n; ++ii)
        std::printf("%.2f ", keep_fraction[ii]);
    std::printf("\n");
    
    std::vector<double> ra_tmp, dec_tmp, z_tmp;

    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    for (size_t ii=0; ii<ra_vec.size(); ++ii)
    {
        size_t idx;
        gsl_histogram_find(sim_z_hist, z_vec[ii], &idx);
        if (gsl_rng_uniform(rng) < keep_fraction[idx])
        {
            ra_tmp.push_back(ra_vec[ii]);
            dec_tmp.push_back(dec_vec[ii]);
            z_tmp.push_back(z_vec[ii]);
        }
    }

    // clean up
    gsl_histogram_free(sim_z_hist);
    gsl_rng_free(rng);

    // assign output
    ra_vec = ra_tmp;
    dec_vec = dec_tmp;
    z_vec = z_tmp;
}

struct GalHelper
{
    // we store the coordinates so we have more efficient lookup hopefully
    // (at the cost of more memory but should be fine)
    double ra, dec, z;
    int64_t hp_idx;
    size_t vec_idx;
    pointing ang;

    GalHelper (size_t vec_idx_, double ra_, double dec_, double z_, T_Healpix_Base<int64_t> &hp_base) :
        ra {ra_}, dec {dec_}, z{z_}, vec_idx {vec_idx_}
    {
        double theta = (90.0-dec) * M_PI/180.0;
        double phi = ra * M_PI/180.0;
        ang = pointing(theta, phi);
        hp_idx = hp_base.ang2pix(ang);
    };
};

inline long double hav (long double theta)
{
    return gsl_pow_2(std::sin(0.5*theta));
}

inline long double haversine (const pointing &a1, const pointing &a2)
{
    long double t1=a1.theta, t2=a2.theta, p1=a1.phi, p2=a2.phi;
    return hav(t1-t2)
               + hav(p1-p2) 
                 * ( - hav(t1-t2) + hav(t1+t2) );
                 // I believe this is the correct modification
}

void fibcoll (std::vector<double> &ra_vec, std::vector<double> &dec_vec, std::vector<double> &z_vec)
{
    // both figures from Chang
    static const long double angscale = 0.01722; // in degrees
    static const double collrate = 0.6;

    // for sampling from overlapping regions according to collision rate
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

    // for HOPEFULLY efficient nearest neighbour search
    static const int64_t nside = 128; // can be tuned, with 128 pixarea/discarea~225
    T_Healpix_Base hp_base (nside, RING, SET_NSIDE);

    std::vector<GalHelper> all_vec;
    for (size_t ii=0; ii<ra_vec.size(); ++ii)
        all_vec.emplace_back(ii, ra_vec[ii], dec_vec[ii], z_vec[ii], hp_base);

    // empty the output arrays
    ra_vec.clear();
    dec_vec.clear();
    z_vec.clear();

    // sort these for efficient access with increasing healpix index
    std::sort(all_vec.begin(), all_vec.end(),
              [](const GalHelper &a, const GalHelper &b){ return a.hp_idx<b.hp_idx; });

    // again, for efficient access we compute the ranges
    // As we only cover a small sky area, a map is probably more efficient than an array
    std::map<int64_t, std::pair<size_t, size_t>> ranges;
    int64_t current_hp_idx = all_vec[0].hp_idx;
    size_t range_start = 0;
    for (size_t ii=0; ii<all_vec.size(); ++ii)
    {
        const auto &g = all_vec[ii];
        if (g.hp_idx != current_hp_idx)
        {
            ranges[current_hp_idx] = std::pair<size_t, size_t>(range_start, ii);
            range_start = ii;
        }
    }

    // we assume the inputs have sufficient entropy that always removing the first galaxy is fine

    // allocate for efficiency
    rangeset<int64_t> query_result;
    std::vector<int64_t> query_vector;

    for (const auto &g : all_vec)
    {
        hp_base.query_disc(g.ang, angscale*M_PI/180.0, query_result);
        query_result.toVector(query_vector);
        bool collided = false;
        for (auto hp_idx : query_vector)
        {
            auto this_range = ranges[hp_idx];
            for (size_t ii=this_range.first; ii<this_range.second; ++ii)
                if (haversine(g.ang, all_vec[ii].ang) < hav(angscale*M_PI/180.0)
                    && g.vec_idx != all_vec[ii].vec_idx)
                // use the fact that haversine is monotonic to avoid inverse operation
                // also check for identity
                {
                    collided = true;
                    break;
                }
            if (collided) break;
        }

        // TODO it could also make sense to call the rng every time we have a collision
        //      in the above loop instead. Maybe not super important though.
        if (collided && gsl_rng_uniform(rng)<collrate) continue;

        // no collision, let's keep this galaxy
        ra_vec.push_back(g.ra);
        dec_vec.push_back(g.dec);
        z_vec.push_back(g.z);
    }

    // for debugging
    std::printf("fiber collision rate: %.2f percent\n",
                100.0*(double)(all_vec.size()-z_vec.size())/(double)(all_vec.size()));
}

template<typename T>
inline double per_unit (T x, double BoxSize)
{
    double x1 = (double)x / BoxSize;
    x1 = std::fmod(x, 1.0);
    return (x1<0.0) ? x1+1.0 : x1;
}

int main (int argc, char **argv)
{
    char **c = argv + 1;

    char *inpath = *(c++);
    char *inident = *(c++);
    char *outident = *(c++);
    double BoxSize = std::atof(*(c++));
    double Omega_m = std::atof(*(c++));
    double zmin = std::atof(*(c++));
    double zmax = std::atof(*(c++));
    int remap_case = std::atoi(*(c++));
    char *boss_dir = *(c++);
    int veto = std::atoi(*(c++)); // whether to apply veto, removes a bit less than 7% of galaxies

    int Nsnaps = 0;
    double times[64];
    while (*c)
        times[Nsnaps++] = std::atof(*(c++));

    // get increasing redshifts
    std::qsort(times, Nsnaps, sizeof(double), dbl_cmp</*reverse=*/true>);

    double redshifts[Nsnaps];
    for (int ii=0; ii<Nsnaps; ++ii)
        redshifts[ii] = 1.0/times[ii] - 1.0;

    // define the redshift boundaries
    // TODO we can maybe do something more sophisticated here
    double redshift_bounds[Nsnaps+1];
    redshift_bounds[0] = zmin;
    redshift_bounds[Nsnaps] = zmax;
    for (int ii=1; ii<Nsnaps; ++ii)
        redshift_bounds[ii] = 0.5*(redshifts[ii-1]+redshifts[ii]);
    
    // convert to comoving distances
    double chi_bounds[Nsnaps+1];
    comoving(Omega_m, Nsnaps+1, redshift_bounds, chi_bounds);

    // initialize the interpolator getting us from comoving to redshift
    const int Ninterp = 1024;
    double z_interp_min = zmin-0.01, z_interp_max=zmax+0.01;
    double z_interp[Ninterp];
    double chi_interp[Ninterp];
    for (int ii=0; ii<Ninterp; ++ii)
        z_interp[ii] = z_interp_min + (z_interp_max-z_interp_min)*(double)ii/(double)(Ninterp-1);
    comoving(Omega_m, Ninterp, z_interp, chi_interp);
    gsl_interp *z_chi_interp = gsl_interp_alloc(gsl_interp_cspline, Ninterp);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_interp_init(z_chi_interp, chi_interp, z_interp, Ninterp);

    // initialize the transformation
    auto C = cuboid::Cuboid(remaps[remap_case]);

    // initialize the survey footprint
    char mask_fname[512];
    cmangle::MangleMask *ang_mask = cmangle::mangle_new();
    std::sprintf(mask_fname, "%s/%s", boss_dir, ang_mask_fname);
    cmangle::mangle_read(ang_mask, mask_fname);

    // much more efficient
    cmangle::set_pixel_map(ang_mask);

    cmangle::MangleMask *veto_masks[Nveto];
    if (veto)
    {
        for (int ii=0; ii<Nveto; ++ii)
        {
            veto_masks[ii] = cmangle::mangle_new();
            std::sprintf(mask_fname, "%s/%s", boss_dir, veto_fnames[ii]);
            cmangle::mangle_read(veto_masks[ii], mask_fname);
            cmangle::set_pixel_map(veto_masks[ii]);
        }
    }

    // the target redshift distribution
    std::vector<double> boss_z = read_boss_z(boss_dir);
    gsl_histogram *boss_z_hist = gsl_histogram_alloc(N_zbins);
    gsl_histogram_set_ranges_uniform(boss_z_hist, zmin, zmax);
    for (auto z : boss_z)
    {
        if (z<zmin || z>zmax) continue;
        gsl_histogram_increment(boss_z_hist, z);
    }

    // this is the output
    std::vector<double> ra_out, dec_out, z_out;

    for (int ii=0; ii<Nsnaps; ++ii)
    {
        char fname[512];
        std::sprintf(fname, "%s/galaxies/galaxies_%s_%.4f.bin", inpath, inident, times[ii]);

        auto fp = std::fopen(fname, "rb");

        // figure out number of galaxies
        std::fseek(fp, 0, SEEK_END);
        auto nbytes = std::ftell(fp);

        auto Ngal = (nbytes/sizeof(float)-1/*rsd factor*/)/6;
        if (!((6*Ngal+1)*sizeof(float)==nbytes)) return 1;

        // go back to beginning
        std::fseek(fp, 0, SEEK_SET);

        float rsd_factor_f;
        std::fread(&rsd_factor_f, sizeof(float), 1, fp);

        float *xgal_f = (float *)std::malloc(3 * Ngal * sizeof(float));
        float *vgal_f = (float *)std::malloc(3 * Ngal * sizeof(float));
        std::fread(xgal_f, sizeof(float), 3*Ngal, fp);
        std::fread(vgal_f, sizeof(float), 3*Ngal, fp);

        // these will contain the outputs
        double *xgal = (double *)std::malloc(3 * Ngal * sizeof(double));
        double *vgal = (double *)std::malloc(3 * Ngal * sizeof(double));

        // now transform the positions and velocities
        for (size_t jj=0; jj<Ngal; ++jj)
        {
            C.Transform(per_unit(xgal_f[3*jj+0], BoxSize),
                        per_unit(xgal_f[3*jj+1], BoxSize),
                        per_unit(xgal_f[3*jj+2], BoxSize),
                        xgal[3*jj+0], xgal[3*jj+1], xgal[3*jj+2]);
            for (int kk=0; kk<3; ++kk) xgal[3*jj+kk] *= BoxSize;

            C.VelocityTransform(vgal_f[3*jj+0], vgal_f[3*jj+1], vgal_f[3*jj+2],
                                vgal[3*jj+0], vgal[3*jj+1], vgal[3*jj+2]);
        }

        // now perform RSD
        double L[] = { C.L1, C.L2, C.L3 };
        for (size_t jj=0; jj<Ngal; ++jj)
        {
            // compute the line-of-sight vector
            double los[3];
            for (int kk=0; kk<3; ++kk) los[kk] = xgal[3*jj+kk] - origin[kk]*BoxSize*L[kk];

            // compute length of the line-of-sight vector
            double abs_los = std::hypot(los[0], los[1], los[2]);

            // compute the velocity projection onto the line of sight
            double vproj = (los[0]*vgal[3*jj+0]+los[1]*vgal[3*jj+1]+los[2]*vgal[3*jj+2])
                           / abs_los;

            for (int kk=0; kk<3; ++kk) xgal[3*jj+kk] += rsd_factor_f * vproj * los[kk] / abs_los;
        }
        
        // choose the galaxies within this redshift shell
        for (size_t jj=0; jj<Ngal; ++jj)
        {
            double los[3];
            for (int kk=0; kk<3; ++kk) los[kk] = xgal[3*jj+kk] - origin[kk]*BoxSize*L[kk];
            
            double chi = std::hypot(los[0], los[1], los[2]);

            if (chi>chi_bounds[ii] && chi<chi_bounds[ii+1])
            // we are in the comoving shell that's coming from this snapshot
            {
                // rotate the line of sight into the NGC footprint and transpose the axes into
                // canonical order
                double x1, x2, x3;
                x1 = std::cos(alpha) * los[2] - std::sin(alpha) * los[1];
                x2 = los[0];
                x3 = std::sin(alpha) * los[2] + std::cos(alpha) * los[1];

                double z = gsl_interp_eval(z_chi_interp, chi_interp, z_interp, chi, acc);
                double theta = std::acos(x3/chi);
                double phi = std::atan2(x2, x1);

                double dec = 90.0-theta/M_PI*180.0;
                double ra = phi/M_PI*180.0;
                if (ra<0.0) ra += 360.0;
                ra += beta;

                // for the angular mask
                cmangle::Point pt;
                cmangle::point_set_from_radec(&pt, ra, dec);
                int64_t poly_id; long double weight;
                cmangle::mangle_polyid_and_weight_pix(ang_mask, &pt, &poly_id, &weight);
                if (weight==0.0L) continue;

                bool vetoed = false;
                if (veto)
                    for (int kk=0; kk<Nveto && !vetoed; ++kk)
                    {
                        cmangle::mangle_polyid_and_weight_pix(veto_masks[kk], &pt, &poly_id, &weight);
                        if (weight!=0.0L) vetoed = true;
                    }
                if (vetoed) continue;

                z_out.push_back(z);
                dec_out.push_back(dec);
                ra_out.push_back(ra);
            }
        }

        std::printf("Done with %d\n", ii);
    }

    // first downsampling before fiber collisions are applied
    downsample(boss_z_hist, fibcoll_rate, ra_out, dec_out, z_out);

    // apply fiber collisions
    fibcoll(ra_out, dec_out, z_out); 

    // now downsample to our final density
    downsample(boss_z_hist, 0.0, ra_out, dec_out, z_out);

    // output
    char fname[512];
    std::sprintf(fname, "%s/galaxies/lightcone_%s_%s.txt", inpath, inident, outident);
    auto fp = std::fopen(fname, "w");
    std::fprintf(fp, "# RA, DEC, z\n");
    for (size_t ii=0; ii<z_out.size(); ++ii)
        std::fprintf(fp, "%.8f %.8f %.8f\n", ra_out[ii], dec_out[ii], z_out[ii]);

    // clean up
    gsl_interp_free(z_chi_interp);
    gsl_interp_accel_free(acc);

    return 0;
}
