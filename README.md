

This is a messy repository with many historical artifacts.
Most things are very specialized and probably not useful.

Below an almost complete directory of the code that was ultimately used.


## Simulations

Input file templates:
* `reps.ini`: REPS parameter file
* `fastpm_script.lua`: FastPM parameter file
* `rockstar_server.cfg`: Rockstar parameter file

These have place holders to be populated by scripts.

Driver code:
* `run_class_reps.sh`
* `run_fastpm.sh`
* `run_rockstar.sh`, `run_rockstar_batch.sh`
* `run_parents.sh`, `run_parents_distributed.sh`
* `jobstep_fastpm.sbatch`, `jobstep_rockstar.sbatch`, `jobstep_rockstar_leftover.sbatch`, `jobstep_parents.sbatch`

These also have place holders, to be populated by a job preparation script.

Quasi-random prior sampling and fiducial cosmology:
* `sample_prior.c`
* `mu_cov_plikHM_TTTEEE_lowl_lowE.dat`, `mnu_prior.dat`
* `draw_cosmo.sh`
* `cosmo_fiducial.sh`

Job preparation and submission (includes chaining to make sure we don't run out of disk space):
* `timesteps.c`
* `prepare_job.sh`, `prepare_job_fiducial.sh`, `prepare_job_varied.sh`
* `submit.sh`, `submit_batch.sh`, `submit_batch_fiducial.sh`, `submit_batch_varied.sh`

Some utilities:
* `reparameterize.py`
* `cmass.sh`, `globals.sh`, `utils.sh`


## Lightcones

HOD prior:
* `hod_prior.dat`, `hod_deriv_prior_v0.dat`

Make galaxies (implementation currently not provided):
* `hod_galaxies.py`
* ..

Make lightcones:
* `lightcone.cpp`
* `lightcone.sh`
* `job_lightcone.sh`
* ...

The code in the `cuboidremap-1.0`, `healpix_lite`, `pymangle` repositories
is almost a direct copy of the respective packages, with only small tweaks.
`healpix_lite` is a diet version of HEALPIX which only implements the functionality
in `healpix_base.h`. This gets rid of most dependencies and greatly simplifies the build
process.


## Summary statistics

Anything that has `vide`, `plk`, `vg(_)plk` in the file name.


## Organization

Working with 200k lightcones and summary statistics files is not easy.
Just locating all valid files is a compute-intensive process.
Furthermore, when measuring summary statistics it is useful to have a central
place where we record which lightcones have already been processed.
This necessitates atomic read+write operations.

It turns out that MySQL (or pretty much any other data base) is a pretty useful tool
for these sorts of operations. In particular, it enables atomic read+write and
efficient queries. We can also easily inspect progress of computations through the
data base.

Some code related to organizing:
* `README_mysql.txt`: summarizing what I figured out about how to install mysql on the Tiger cluster
* `mysql_driver.c`: the main driver code
* `valid_outputs.c`
* `expand_nodelist.c`: a small tool to convert the particular format of node lists used by slurm into
   a simple list. Needed to get around a bug in a particular version of MPI.
* `collect_*.py`


## Inference

* `derivatives.py`
* `datavector.py`
* `cut.py`, `compress.py`, `cut_compress.py`, `make_compression.py`
* `read_txt.py`
* `lfi_train.py`
* `lfi_load_posterior.py`
* `lfi_sample.py`, `lfi_sample_for_coverage.py`
* `coverage.py`

There are some historical files related to initial attempts with likelihood-based inference
as well as DeepSet (on void catalogs) for maximum information.
Some of this code can be located by matching `*emulator*.py`.


## EFTofLSS

* `boss_full_shape_ours*`


## Plotting

All final plotting utilities match `_plot_*.py`
