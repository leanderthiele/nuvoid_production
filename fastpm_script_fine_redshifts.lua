-- NOTE can find all these options in fastpm/src/lua-runtime-fastpm.lua
--      with some explanations



-- simulation size
nc = <<<nc>>>
boxsize = <<<boxsize>>>

-- time sequence (simulation steps)
-- I believe these are pretty similar to what was used in Bayer+2020

-- log=5 and lin=20 are Bayer+2020
n_steps_log = 5
n_steps_lin = 20
z_i = 99
z_m = 19
a_i = 1. / (1. + z_i)
a_m = 1. / (1. + z_m)
a_f = 0.6667

-- time_step = loglinspace(a_i, a_m, a_f, n_steps_log, n_steps_lin)
time_step = { 0.01, 0.01495349, 0.02236068, 0.03343702, 0.05,
              0.08854375, 0.1270875, 0.16563125, 0.204175,
              0.24271875, 0.2812625, 0.31980625, 0.35835, 0.39689375,
              0.4354375, 0.47398125, 0.512525, 0.55106875,
              0.571428571, 0.579710145, 0.588235294, 0.597014925,
              0.60606061, 0.615384615, 0.625, 0.634920635,
              0.64516129, 0.677737705, 0.6667, 0.67796602,
              0.6898 }
-- take the last time-step a bit larger than the final snapshot we want,
--   otherwise FastPM doesn't store the snapshot

-- write outputs here
output_redshifts = {<<<OUT_REDSHIFTS>>>}

-- LambdaCDM
-- needs to match the reps.ini file obviously
Omega_m = <<<Omega_m>>> -- this is (cdm + baryon + ncdm), so includes the neutrinos
h = <<<h>>>
T_cmb = 2.725

-- Perturbations, i.e. REPS output
REPS_OUTPUT = "<<<REPS_OUTPUT>>>"
read_powerspectrum = REPS_OUTPUT .. "/Pcb_rescaled_z99.0000.txt"
read_linear_growth_rate = REPS_OUTPUT .. "/fcb_z99.0000.txt"

-- STARTNEUTRINOS
N_eff = <<<N_eff>>>
N_nu = <<<N_nu>>>
m_ncdm = {<<<m_nu>>>} -- comma separated individual masses in eV
n_shell = 10
ncdm_sphere_scheme = "fibonacci"
n_side = 3 -- maybe worth playing with (e.g. 10, 20 -- but only on pretty small scales)
every_ncdm = 4 -- increase to have fewer neutrinos relative to CDM, 4 is the Bayer+2020 choice
               -- consider setting to 1
lvk = true

-- these for having neutrinos in the background
ncdm_freestreaming = false
ncdm_matterlike = false

-- the neutrino growth
read_powerspectrum_ncdm = REPS_OUTPUT .. "/Pn_rescaled_z99.0000.txt"
read_linear_growth_rate_ncdm = REPS_OUTPUT .. "/fn_z99.0000.txt"
linear_density_redshift_ncdm = z_i
-- ENDNEUTRINOS

-- where these files have been computed
linear_density_redshift = z_i

-- initial conditions?
random_seed = <<<random_seed>>>
particle_fraction = 1.0

-- code options
force_mode = "fastpm"
pm_nc_factor = {{0.0, 1}, {0.0001, 2}} -- list of (a, PM resolution)
remove_cosmic_variance = false -- this is the paired-and-fixed approach I think, which doesn't
                               -- make much of a difference on sub-BAO scales, so ignore
growth_mode = "ODE" -- ODE is the correct choice here
za = true -- this flag switches between Zeldovich and 2LPT initial conditions
           -- I believe Bayer+2020 used za=true and maybe we should do this too,
           -- because it seems to be a bit faster than 2LPT
np_alloc_factor = 2.0 -- TODO I think the usual choice is 4, but maybe we can get the memory
                      --      down without sacrificing performance
                      -- UPDATE yes it seems like we can decrease this to 2.0 and still get
                      --        the same runtime
kernel_type = "3_4" -- default: 1_4, apparently 3_4 gives more low-mass halos

-- output
FASTPM_OUTPUT = "<<<FASTPM_OUTPUT>>>"
sort_snapshot = false -- sort snaps by ID, default is true but needs a lot of communication
write_snapshot = FASTPM_OUTPUT .. "/snap"
write_powerspectrum = FASTPM_OUTPUT .. "/powerspectra/powerspectrum"

-- TODO running into SIGSEGV when RFOF is on...
write_fof = FASTPM_OUTPUT .. "/fof"
-- write_rfof = FASTPM_OUTPUT .. "/rfof" -- TODO =relaxed-FOF, what is this?

-- fof parameters
fof_linkinglength = 0.2 -- default: 0.2
fof_nmin = 20 -- default: 20
