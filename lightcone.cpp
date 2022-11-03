#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <functional>
#include <map>
#include <utility>

#include <omp.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_statistics_double.h>

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
                    // 1.0000 1.0000 1.0000
                    // trivial case, only for debugging
                    { 1, 0, 0,
                      0, 1, 0,
                      0, 0, 1 },
                  };

// get from the quadrant ra=[-90,90], dec=[0,90] to the NGC footprint
// we only need a rotation around the y-axis I believe
const double alpha = 97.0 * M_PI / 180.0; // rotation around y-axis
const double beta = 6.0; // rotation around z-axis, in degrees

// before initial downsampling, we increase the fiber collision rate by this
// to make sure that after fiber collisions we still have enough galaxies in each redshift bin
const double fibcoll_rate_correction = 0.05;

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

// how many interpolation stencils we use to get from chi to z
const int N_interp = 1024;

// ---- GLOBAL VARIABLES -----

// these are the input variables, populated from command line
const char *inpath, *inident, *outident, *boss_dir;
double BoxSize, Omega_m, zmin, zmax;
int remap_case, Nsnaps;
bool correct, veto, stitch_before_RSD, verbose;
unsigned augment;

// describe the available snapshots and redshift stitching
std::vector<double> snap_times, snap_redshifts, snap_chis, redshift_bounds, chi_bounds;

// utilities for converting between redshift and comoving distance
gsl_spline *z_chi_interp; gsl_interp_accel *z_chi_interp_acc;

// describe the remapping
cuboid::Cuboid C;
double Li[3]; // the sidelengths in decreasing order

// describe the survey footprint
cmangle::MangleMask *ang_mask;
cmangle::MangleMask *veto_masks[Nveto]; // maybe unused

// contains the measured redshift distribution used to downsample the theory catalog
gsl_histogram *boss_z_hist;

// these are our outputs
std::vector<double> RA, DEC, Z;

// ----- ROUTINES USED IN MAIN -------
// due to laziness, we rely on global variables, so the order of calls
// is important...

void process_args (int argc, const char **argv);

// downsamples to the boss_z_hist, up to plus_factor
void downsample (double plus_factor);

// modifies RA, DEC, Z
// if only_measure=false,
// returns the fiber collision rate
template<bool only_measure>
double fibcoll (void);

// sets times, redshifts, redshift_bounds, chi_bounds
void process_times (void);

// initialize the interpolator getting us from comoving to redshift
void interpolate_chi_z (void);

// initialize the mangle masks
void init_masks (void);

// get the target redshift distribution
void measure_boss_nz (void);

void read_snapshot (int snap_idx,
                    std::vector<float> &xgal_f, std::vector<float> &vgal_f, std::vector<float> &vhlo_f,
                    size_t &Ngal);

void remap_snapshot (size_t Ngal,
                     const std::vector<float> &xgal_f,
                     const std::vector<float> &vgal_f,
                     const std::vector<float> &vhlo_f,
                     std::vector<double> &xgal,
                     std::vector<double> &vgal,
                     std::vector<double> &vhlo);

// not used anymore as a separate routine, we do it within choose_galaxies
// void RSD (int snap_idx, size_t Ngal, std::vector<double> &xgal, const std::vector<double> &vgal);

// this routine modifies RA, DEC, Z
void choose_galaxies (int snap_idx, size_t Ngal,
                      const std::vector<double> &xgal,
                      const std::vector<double> &vgal,
                      const std::vector<double> &vhlo);

// writes in the VIDE format
void write_to_disk (void);

int main (int argc, const char **argv)
{
    process_args(argc, argv);

    if (verbose) std::printf("process_times\n");
    process_times();

    z_chi_interp = gsl_spline_alloc(gsl_interp_cspline, N_interp);
    z_chi_interp_acc = gsl_interp_accel_alloc();
    if (verbose) std::printf("interpolate_chi_z\n");
    interpolate_chi_z();

    // initialize the transformation
    C = cuboid::Cuboid(remaps[remap_case]);
    Li[0] = C.L1; Li[1] = C.L2; Li[2] = C.L3;

    // initialize survey footprint and veto masks (if requested)
    ang_mask = cmangle::mangle_new();
    if (veto) for (int ii=0; ii<Nveto; ++ii) veto_masks[ii] = cmangle::mangle_new();
    if (verbose) std::printf("init_masks\n");
    init_masks();

    // the target redshift distribution
    if (verbose) std::printf("measure_boss_nz\n");
    measure_boss_nz();

    #pragma omp parallel for
    for (int ii=0; ii<Nsnaps; ++ii)
    {
        size_t Ngal;

        // contains the 32bit data from disk
        std::vector<float> xgal_f, vgal_f, vhlo_f;
        if (verbose) std::printf("\tread_snapshot\n");
        read_snapshot(ii, xgal_f, vgal_f, vhlo_f, Ngal);

        // these will contain the outputs of the remapping
        std::vector<double> xgal, vgal, vhlo;
        if (verbose) std::printf("\tremap_snapshot\n");
        remap_snapshot(Ngal, xgal_f, vgal_f, vhlo_f, xgal, vgal, vhlo);

        // choose the galaxies within this redshift shell
        // This routine also implements lightcone correction
        // and RSD
        if (verbose) std::printf("\tchoose_galaxies\n");
        choose_galaxies(ii, Ngal, xgal, vgal, vhlo);

        if (verbose) std::printf("Done with %d\n", ii);
    }
    
    // get an idea of the fiber collision rate in our sample,
    // so the subsequent downsampling is to the correct level.
    // This is justified as all this stuff is not super expensive.
    double fibcoll_rate = fibcoll</*only_measure=*/true>();

    // first downsampling before fiber collisions are applied
    if (verbose) std::printf("downsample\n");
    downsample(fibcoll_rate+fibcoll_rate_correction);

    // apply fiber collisions
    if (verbose) std::printf("fibcoll\n");
    fibcoll</*only_measure=*/false>(); 

    // now downsample to our final density
    if (verbose) std::printf("downsample\n");
    downsample(0.0);

    // output
    if (verbose) std::printf("write_to_disk\n");
    write_to_disk();

    // clean up
    gsl_spline_free(z_chi_interp); gsl_interp_accel_free(z_chi_interp_acc);
    gsl_histogram_free(boss_z_hist);
    cmangle::mangle_free(ang_mask);
    if (veto) for (int ii=0; ii<Nveto; ++ii) cmangle::mangle_free(veto_masks[ii]);

    return 0;
}

// ------ IMPLEMENTATION ------

void process_args (int argc, const char **argv)
{
    const char **c = argv + 1;

    if (!*c) // can call without arguments to get usage information
    {
        std::printf("COMMAND LINE ARGUMENTS:\n"
                    "\tinpath\n"
                    "\tinident\n"
                    "\toutident\n"
                    "\tBoxSize\n"
                    "\tOmega_m\n"
                    "\tzmin\n"
                    "\tzmax\n"
                    "\tremap_case\n"
                    "\tcorrect\n"
                    "\taugment\n"
                    "\tboss_dir\n"
                    "\tveto\n"
                    "\tstitch_before_RSD\n"
                    "\tverbose\n"
                    "\tsnap_times[...]\n");
        throw std::runtime_error("Invalid arguments");
    }

    inpath = *(c++);
    inident = *(c++);
    outident = *(c++);
    BoxSize = std::atof(*(c++));
    Omega_m = std::atof(*(c++));
    zmin = std::atof(*(c++));
    zmax = std::atof(*(c++));
    remap_case = std::atoi(*(c++));
    correct = std::atoi(*(c++));
    augment = std::atoi(*(c++));
    boss_dir = *(c++);
    veto = std::atoi(*(c++)); // whether to apply veto, removes a bit less than 7% of galaxies
    stitch_before_RSD = std::atoi(*(c++));
    verbose = std::atoi(*(c++));

    while (*c) snap_times.push_back(std::atof(*(c++)));
    Nsnaps = snap_times.size();

    assert((c-argv)==argc);
}

double comoving_integrand (double z, void *p)
{
    static const double H0inv = 2.99792458e3; // Mpc/h
    double Omega_m = *(double *)p;
    return H0inv / std::sqrt( Omega_m*gsl_pow_3(1.0+z)+(1.0-Omega_m) );
}

void comoving (int N, double *z, double *chi)
// populate chi with chi(z) in Mpc/h
{
    gsl_function F;
    F.function = comoving_integrand;
    F.params = &Omega_m;

    static const size_t ws_size = 1024;
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(ws_size);

    double err;
    for (int ii=0; ii<N; ++ii)
        gsl_integration_qag(&F, 0.0, z[ii], 1e-2, 0.0, ws_size, 6, ws, chi+ii, &err);

    gsl_integration_workspace_free(ws);
}

void process_times (void)
{
    // get increasing redshifts
    std::sort(snap_times.begin(), snap_times.end(), [](double a, double b){ return a>b; });

    for (int ii=0; ii<Nsnaps; ++ii) snap_redshifts.push_back(1.0/snap_times[ii] - 1.0);

    snap_chis.resize(Nsnaps);
    comoving(Nsnaps, snap_redshifts.data(), snap_chis.data());

    // define the redshift boundaries
    // TODO we can maybe do something more sophisticated here
    redshift_bounds.push_back(zmin);
    for (int ii=1; ii<Nsnaps; ++ii)
        redshift_bounds.push_back(0.5*(snap_redshifts[ii-1]+snap_redshifts[ii]));
    redshift_bounds.push_back(zmax);
    
    // convert to comoving distances
    chi_bounds.resize(Nsnaps+1);
    comoving(Nsnaps+1, redshift_bounds.data(), chi_bounds.data());
}

void interpolate_chi_z (void)
{
    double z_interp_min = zmin-0.01, z_interp_max=zmax+0.01;
    double *z_interp = (double *)std::malloc(N_interp * sizeof(double));
    double *chi_interp = (double *)std::malloc(N_interp * sizeof(double));
    for (int ii=0; ii<N_interp; ++ii)
        z_interp[ii] = z_interp_min + (z_interp_max-z_interp_min)*(double)ii/(double)(N_interp-1);
    comoving(N_interp, z_interp, chi_interp);
    gsl_spline_init(z_chi_interp, chi_interp, z_interp, N_interp);
    std::free(z_interp); std::free(chi_interp);
}

void init_masks (void)
{
    char mask_fname[512];
    std::sprintf(mask_fname, "%s/%s", boss_dir, ang_mask_fname);
    cmangle::mangle_read(ang_mask, mask_fname);
    cmangle::set_pixel_map(ang_mask);

    if (veto)
        // this stuff is pretty expensive so do it in parallel
        #pragma omp parallel for
        for (int ii=0; ii<Nveto; ++ii)
        {
            char veto_fname[512];
            std::sprintf(veto_fname, "%s/%s", boss_dir, veto_fnames[ii]);
            cmangle::mangle_read(veto_masks[ii], veto_fname);
            cmangle::set_pixel_map(veto_masks[ii]);
        }
}

void measure_boss_nz (void)
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
    std::vector<double> boss_z;
    boss_z.resize(Ngal);
    std::fread(boss_z.data(), sizeof(double), Ngal, fp);
    std::fclose(fp);

    // remove values outside our bounds
    std::vector<double> boss_z_cleaned;
    std::copy_if(boss_z.begin(), boss_z.end(), std::back_inserter(boss_z_cleaned),
                 [](double this_z){ return this_z>zmin && this_z<zmax; });

    // measure the standard deviation for Scott's rule
    double sd = gsl_stats_sd(boss_z_cleaned.data(), 1, boss_z_cleaned.size());

    // Scott's rule
    double dz = 3.5 * sd / std::cbrt((double)boss_z_cleaned.size());
    size_t Nbins = std::round((zmax-zmin)/dz);

    boss_z_hist = gsl_histogram_alloc(Nbins);
    gsl_histogram_set_ranges_uniform(boss_z_hist, zmin, zmax);

    std::for_each(boss_z_cleaned.begin(), boss_z_cleaned.end(),
                  [](double this_z){ gsl_histogram_increment(boss_z_hist, this_z); });
}

void read_snapshot (int snap_idx,
                    std::vector<float> &xgal_f, std::vector<float> &vgal_f, std::vector<float> &vhlo_f,
                    size_t &Ngal)
{
    char fname[512];
    std::sprintf(fname, "%s/galaxies%s%s_%.4f.bin",
                        inpath, (std::strlen(inident)) ? "_" : "", inident, snap_times[snap_idx]);

    auto fp = std::fopen(fname, "rb");

    // figure out number of galaxies
    std::fseek(fp, 0, SEEK_END);
    auto nbytes = std::ftell(fp);

    Ngal = nbytes/sizeof(float)/9;
    if (!(9*Ngal*sizeof(float)==nbytes)) throw std::runtime_error("failed in read_snapshot");

    // go back to beginning
    std::fseek(fp, 0, SEEK_SET);

    xgal_f.resize(3 * Ngal);
    vgal_f.resize(3 * Ngal);
    vhlo_f.resize(3 * Ngal);
    std::fread(xgal_f.data(), sizeof(float), 3*Ngal, fp);
    std::fread(vgal_f.data(), sizeof(float), 3*Ngal, fp);
    std::fread(vhlo_f.data(), sizeof(float), 3*Ngal, fp);

    // now we can close
    std::fclose(fp);
}

template<typename T>
inline double per_unit (T x)
{
    double x1 = (double)x / BoxSize;
    x1 = std::fmod(x1, 1.0);
    return (x1<0.0) ? x1+1.0 : x1;
}

inline void reflect (unsigned r, double *x, double *v, double *vh)
{
    for (unsigned ii=0; ii<3; ++ii)
        if (r & (1<<ii))
        {
            x[ii] = BoxSize - x[ii];
            v[ii] *= -1.0;
            if (correct) vh[ii] *= -1.0;
        }
}

inline void transpose (unsigned t, double *x)
{
    if      (t==0) return;
    else if (t==1) std::swap(x[0], x[1]);
    else if (t & 1)
    {
        std::swap(x[0], x[2]);
        if (t & 4) std::swap(x[0], x[1]);
    }
    else
    {
        std::swap(x[1], x[2]);
        if (t & 4) std::swap(x[0], x[1]);
    }
}

void remap_snapshot (size_t Ngal,
                     const std::vector<float> &xgal_f,
                     const std::vector<float> &vgal_f,
                     const std::vector<float> &vhlo_f,
                     std::vector<double> &xgal,
                     std::vector<double> &vgal,
                     std::vector<double> &vhlo)
{
    xgal.resize(3 * Ngal); vgal.resize(3 * Ngal);
    if (correct) vhlo.resize(3 * Ngal);

    unsigned r = augment / 6; // labels the 8 reflection cases
    unsigned t = augment % 6; // labels the 6 transposition cases

    for (size_t ii=0; ii<Ngal; ++ii)
    {
        double x[3], v[3], vh[3];
        for (int kk=0; kk<3; ++kk)
        {
            x[kk] = xgal_f[3*ii+kk];
            v[kk] = vgal_f[3*ii+kk];
            vh[kk] = vhlo_f[3*ii+kk];
        }

        reflect(r, x, v, vh);
        transpose(t, x); transpose(t, v);
        if (correct) transpose(t, vh);

        C.Transform(per_unit(x[0]), per_unit(x[1]), per_unit(x[2]),
                    xgal[3*ii+0], xgal[3*ii+1], xgal[3*ii+2]);
        for (int kk=0; kk<3; ++kk) xgal[3*ii+kk] *= BoxSize;

        C.VelocityTransform(v[0], v[1], v[2],
                            vgal[3*ii+0], vgal[3*ii+1], vgal[3*ii+2]);

        if (correct)
            C.VelocityTransform(vh[0], vh[1], vh[2],
                                vhlo[3*ii+0], vhlo[3*ii+1], vhlo[3*ii+2]);
    }
}

/* not used anymore
void RSD (int snap_idx, size_t Ngal, std::vector<double> &xgal, const std::vector<double> &vgal)
{
    double rsd_factor = (1.0+snap_redshifts[snap_idx])
                        / ( 100.0 * std::sqrt(Omega_m * gsl_pow_3(1.0+snap_redshifts[snap_idx]) + (1.0-Omega_m)) );
    for (size_t jj=0; jj<Ngal; ++jj)
    {
        // compute the line-of-sight vector
        double los[3];
        for (int kk=0; kk<3; ++kk) los[kk] = xgal[3*jj+kk] - origin[kk]*BoxSize*Li[kk];

        // compute length of the line-of-sight vector
        double abs_los = std::hypot(los[0], los[1], los[2]);

        // compute the velocity projection onto the line of sight
        double vproj = (los[0]*vgal[3*jj+0]+los[1]*vgal[3*jj+1]+los[2]*vgal[3*jj+2])
                       / abs_los;

        for (int kk=0; kk<3; ++kk)
            xgal[3*jj+kk] += rsd_factor * vproj * los[kk] / abs_los;
    }
}
*/

void choose_galaxies (int snap_idx, size_t Ngal,
                      const std::vector<double> &xgal,
                      const std::vector<double> &vgal,
                      const std::vector<double> &vhlo)
{
    // h km/s/Mpc
    const double Hz = 100.0 * std::sqrt(Omega_m*gsl_pow_3(1.0+snap_redshifts[snap_idx])+(1.0-Omega_m));

    double rsd_factor = (1.0+snap_redshifts[snap_idx]) / Hz;

    for (size_t jj=0; jj<Ngal; ++jj)
    {
        double los[3];
        for (int kk=0; kk<3; ++kk) los[kk] = xgal[3*jj+kk] - origin[kk]*BoxSize*Li[kk];
        
        double chi = std::hypot(los[0], los[1], los[2]);

        if (correct)
        {
            double vhloproj = (los[0]*vhlo[3*jj+0]+los[1]*vhlo[3*jj+1]+los[2]*vhlo[3*jj+2])
                              / chi;

            // map onto the lightcone -- we assume that this is a relatively small correction
            //                           so we just do it to first order
            double delta_z = (chi - snap_chis[snap_idx]) / (299792.458 + vhloproj) * Hz;

            // correct the position accordingly
            for (int kk=0; kk<3; ++kk) los[kk] -= delta_z * vhlo[3*jj+kk] / Hz;

            // only a small correction I assume (and it is borne out by experiment!)
            chi = std::hypot(los[0], los[1], los[2]);
        }

        double chi_stitch; // the one used for stitching

        if (stitch_before_RSD) chi_stitch = chi;

        // the radial velocity
        double vproj = (los[0]*vgal[3*jj+0]+los[1]*vgal[3*jj+1]+los[2]*vgal[3*jj+2])
                       / chi;

        // now add RSD (the order is important here, as RSD can be a pretty severe effect)
        for (int kk=0; kk<3; ++kk) los[kk] += rsd_factor * vproj * los[kk] / chi;

        // I'm lazy
        chi = std::hypot(los[0], los[1], los[2]);

        if (!stitch_before_RSD) chi_stitch = chi;

        if (chi_stitch>chi_bounds[snap_idx] && chi_stitch<chi_bounds[snap_idx+1]
            // prevent from falling out of the redshift interval we're mapping into
            // (only relevent if stitching based on chi before RSD)
            && (!stitch_before_RSD || (chi>chi_bounds[0] && chi<chi_bounds[Nsnaps])))
        // we are in the comoving shell that's coming from this snapshot
        {
            // rotate the line of sight into the NGC footprint and transpose the axes into
            // canonical order
            double x1, x2, x3;
            x1 = std::cos(alpha) * los[2] - std::sin(alpha) * los[1];
            x2 = los[0];
            x3 = std::sin(alpha) * los[2] + std::cos(alpha) * los[1];

            double z = gsl_spline_eval(z_chi_interp, chi, z_chi_interp_acc);
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
            if (weight==0.0L) goto not_chosen;

            if (veto)
                for (int kk=0; kk<Nveto; ++kk)
                {
                    cmangle::mangle_polyid_and_weight_pix(veto_masks[kk], &pt, &poly_id, &weight);
                    if (weight!=0.0L) goto not_chosen;
                }

            // this is executed in parallel, modifying global variables
            #pragma omp critical (CHOOSE_APPEND)
            {
                RA.push_back(ra);
                DEC.push_back(dec);
                Z.push_back(z);
            }
            continue;

            not_chosen : /* do nothing */;
        }
    }
}

void downsample (double plus_factor)
// downsamples to the boss_z_hist, up to plus_factor
{
    const gsl_histogram *target_hist = boss_z_hist;

    gsl_histogram *sim_z_hist = gsl_histogram_clone(target_hist); // get the same ranges
    gsl_histogram_reset(sim_z_hist);
    for (auto z : Z)
        gsl_histogram_increment(sim_z_hist, z);
    double keep_fraction[target_hist->n];
    for (int ii=0; ii<target_hist->n; ++ii)
    {
        // note the GSL bin counts are doubles already
        keep_fraction[ii] = (1.0+plus_factor)
                            * gsl_histogram_get(target_hist, ii) / gsl_histogram_get(sim_z_hist, ii);

        // give warning if something goes wrong
        if (keep_fraction[ii] > 1.0 && plus_factor==0.0)
            std::fprintf(stderr, "[WARNING] at z=%.4f, do not have enough galaxies, keep_fraction=%.4f\n",
                                 0.5*(target_hist->range[ii]+target_hist->range[ii+1]),
                                 keep_fraction[ii]);
    }

    // for debugging
    if (verbose)
    {
        std::printf("keep_fraction:\n");
        for (int ii=0; ii<target_hist->n; ++ii)
            std::printf("%.2f ", keep_fraction[ii]);
        std::printf("\n");
    }
    
    std::vector<double> ra_tmp, dec_tmp, z_tmp;

    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

    for (size_t ii=0; ii<RA.size(); ++ii)
    {
        size_t idx;
        gsl_histogram_find(sim_z_hist, Z[ii], &idx);
        if (gsl_rng_uniform(rng) < keep_fraction[idx])
        {
            ra_tmp.push_back(RA[ii]);
            dec_tmp.push_back(DEC[ii]);
            z_tmp.push_back(Z[ii]);
        }
    }

    // clean up
    gsl_histogram_free(sim_z_hist);
    gsl_rng_free(rng);

    // assign output
    RA = std::move(ra_tmp);
    DEC = std::move(dec_tmp);
    Z = std::move(z_tmp);
}

struct GalHelper
{
    // we store the coordinates so we have more efficient lookup hopefully
    // (at the cost of more memory but should be fine)
    double ra, dec, z;
    int64_t hp_idx;
    unsigned long id;
    pointing ang;

    GalHelper ( ) = default;

    GalHelper (unsigned long id_, double ra_, double dec_, double z_,
               const T_Healpix_Base<int64_t> &hp_base) :
        ra {ra_}, dec {dec_}, z{z_}, id {id_}
    {
        double theta = (90.0-dec) * M_PI/180.0;
        double phi = ra * M_PI/180.0;
        ang = pointing(theta, phi);
        hp_idx = hp_base.ang2pix(ang);
    };
};

inline long double hav (long double theta)
{
    auto x = std::sin(0.5*theta);
    return x * x;
}

inline long double haversine (const pointing &a1, const pointing &a2)
{
    long double t1=a1.theta, t2=a2.theta, p1=a1.phi, p2=a2.phi;
    return hav(t1-t2) + hav(p1-p2) * ( - hav(t1-t2) + hav(t1+t2) );
}

template<bool only_measure>
double fibcoll ()
{
    // both figures from Chang
    // we are dealing with pretty small angle differences so better do things in long double
    static const long double angscale = 0.01722L * M_PIl / 180.0L; // in rad
    static const double collrate = 0.6;

    // for sampling from overlapping regions according to collision rate
    // and assignment of random galaxy IDs
    std::vector<gsl_rng *> rngs;
    for (int ii=0; ii<omp_get_max_threads(); ++ii)
        rngs.push_back(gsl_rng_alloc(gsl_rng_default));

    // for HOPEFULLY efficient nearest neighbour search
    static const int64_t nside = 128; // can be tuned, with 128 pixarea/discarea~225
    const T_Healpix_Base hp_base (nside, RING, SET_NSIDE);

    // we assign random 64bit IDs to the galaxies
    // This is better than using their index in the input arrays, since these arrays
    // have some ordering (low redshifts go first), and this would slightly bias
    // the fiber collision removal implmented later
    // The collision rate is tiny for 64bit random numbers and thus has no impact.
    std::vector<GalHelper> all_vec { RA.size() };
    #pragma omp parallel
    {
        auto rng = rngs[omp_get_thread_num()];
        
        #pragma omp for
        for (size_t ii=0; ii<RA.size(); ++ii)
            all_vec[ii] = GalHelper(gsl_rng_get(rng), RA[ii], DEC[ii], Z[ii], hp_base);
    }

    if constexpr (!only_measure)
    {
        // empty the output arrays
        RA.clear();
        DEC.clear();
        Z.clear();
    }

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
            current_hp_idx = g.hp_idx;
        }
    }

    // we assume the inputs have sufficient entropy that always removing the first galaxy is fine

    size_t Nkept = 0;

    #pragma omp parallel reduction(+:Nkept)
    {
        // allocate for efficiency, I don't know how rangeset works internally
        // so we do this the old-fashioned way (not private etc)
        rangeset<int64_t> query_result;
        std::vector<int64_t> query_vector;

        // generator used by this thread
        auto rng = rngs[omp_get_thread_num()];

        #pragma omp for schedule (dynamic, 1024)
        for (size_t ii=0; ii<all_vec.size(); ++ii)
        {
            const auto &g = all_vec[ii];

            // one can gain performance here by playing with "fact"
            hp_base.query_disc_inclusive(g.ang, angscale, query_result, /*fact=*/4);
            query_result.toVector(query_vector);

            for (auto hp_idx : query_vector)
            {
                auto this_range = ranges[hp_idx];
                for (size_t ii=this_range.first; ii<this_range.second; ++ii)
                    if (g.id > all_vec[ii].id
                        && haversine(g.ang, all_vec[ii].ang) < hav(angscale))
                    // use the fact that haversine is monotonic to avoid inverse operation
                    // by using the greater-than check, we ensure to remove only one member
                    // of each pair
                        goto collided;
            }
            goto not_collided;

            // TODO it could also make sense to call the rng every time we have a collision
            //      in the above loop instead. Maybe not super important though.
            collided :
            if (gsl_rng_uniform(rng)<collrate) continue;

            not_collided :
            ++Nkept;
            if constexpr (!only_measure)
                // no collision, let's keep this galaxy. We're writing into global variables so need
                // to be careful
                #pragma omp critical (FIBCOLL_APPEND)
                {
                    RA.push_back(g.ra);
                    DEC.push_back(g.dec);
                    Z.push_back(g.z);
                }
        }
    }

    // clean up
    for (int ii=0; ii<rngs.size(); ++ii)
        gsl_rng_free(rngs[ii]);
    
    if constexpr (!only_measure)
        assert(Nkept == RA.size());

    double fibcoll_rate = (double)(all_vec.size()-Nkept)/(double)(all_vec.size());

    return fibcoll_rate;
}


void write_to_disk (void)
{
    char fname[512];
    std::sprintf(fname, "%s/lightcone%s%s_%s.txt",
                        inpath, (std::strlen(inident)) ? "_" : "", inident, outident);
    auto fp = std::fopen(fname, "w");
    for (size_t ii=0; ii<Z.size(); ++ii)
        std::fprintf(fp, "%lu 0 0 %.10f %.10f %.8f 1.0000 0.0\n", ii, RA[ii], DEC[ii], 299792.458*Z[ii]);
}

