-- NOTE can find all these options in fastpm/src/lua-runtime-fastpm.lua
--      with some explanations



-- simulation size
nc = <<<nc>>>
boxsize = <<<boxsize>>>

-- time sequence (simulation steps)
-- I believe these are pretty similar to what was used in Bayer+2020
time_step = { <<<TIME_STEPS>>> }

-- write outputs here
output_redshifts = { <<<OUT_REDSHIFTS>>> }

-- LambdaCDM
-- needs to match the reps.ini file obviously
Omega_m = <<<Omega_m>>> -- this is (cdm + baryon + ncdm), so includes the neutrinos
h = <<<h>>>
T_cmb = 2.725

-- Perturbations, i.e. REPS output. Note the initial redshift is in .4f
REPS_OUTPUT = "<<<REPS_OUTPUT>>>"
read_powerspectrum = REPS_OUTPUT .. "/Pcb_rescaled_z<<<Z_INITIAL>>>.txt"
read_linear_growth_rate = REPS_OUTPUT .. "/fcb_z<<<Z_INITIAL>>>.txt"

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
read_powerspectrum_ncdm = REPS_OUTPUT .. "/Pn_rescaled_z<<<Z_INITIAL>>>.txt"
read_linear_growth_rate_ncdm = REPS_OUTPUT .. "/fn_z<<<Z_INITIAL>>>.txt"
linear_density_redshift_ncdm = <<<Z_INITIAL>>>
-- ENDNEUTRINOS

-- where these files have been computed
linear_density_redshift = <<<Z_INITIAL>>>

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
