import os
import os.path
import sys
from sys import argv
import re
import subprocess
import warnings

import numpy as np

import nbodykit.lab as NBL

from mpi4py import MPI
from plk import PLKCalc

# version of the fiducial HOD
VERSION = int(argv[1])

comm_world = MPI.COMM_WORLD
world_rank = comm_world.Get_rank()

codebase = '/home/lthiele/nuvoid_production'
CPUS_PER_TASK = int(os.environ['CPUS_PER_TASK'])

# get rid of annoying warnings from nbodykit
for w in [np.ComplexWarning, np.VisibleDeprecationWarning, ] :
    warnings.filterwarnings('ignore', category=w)

def report_state(running_idx, seed_idx, lightcone_idx, state) :
    subprocess.run(f'{codebase}/mysql_driver end_fiducials_plk {VERSION} {running_idx} {seed_idx} {lightcone_idx} {state}',
                   shell=True, check=True)

with NBL.TaskManager(cpus_per_task=CPUS_PER_TASK, use_all_cpus=True) as tm :
    
    plk_calc = PLKCalc(comm=tm.comm)

    # we make a fake long list
    for ii in tm.iterate(list(range(10000))) :
        
        # get the sample we should work on
        if tm.comm.rank == 0 :

            result = subprocess.run(f'{codebase}/mysql_driver create_fiducials_plk {VERSION}',
                                    shell=True, capture_output=True, check=True)

            # these are strings!
            running_idx, seed_idx, lightcone_idx, hod_hash = result.stdout.strip().decode().split()
            if int(seed_idx) < 0 :
                # no work left
                continue

            subprocess.run(f'{codebase}/mysql_driver start_fiducials_plk {VERSION} {running_idx} {seed_idx} {lightcone_idx}',
                           shell=True, check=True)

            # this is where we operate
            wrk_dir = f'/scratch/gpfs/lthiele/nuvoid_production/cosmo_fiducial_{seed_idx}/lightcones/{hod_hash}'

            lightcone_file = f'{wrk_dir}/lightcone_{lightcone_idx}.bin'

            if not os.path.isfile(lightcone_file) :
                print(f'{lightcone_file} not found!', file=sys.stderr)
                report_state(running_idx, seed_idx, lightcone_idx, 1)
                continue
        else :
            lightcone_file = None

        # let everyone on the team know which files we should operate on
        lightcone_file = tm.comm.bcast(lightcone_file)

        augment = re.search('(?<=lightcone_)[0-9]*', lightcone_file)[0]

        successful = False
        try :
            plk_result = plk_calc.compute_from_fname(lightcone_file)
            successful = True
        except ValueError as e :
            if 'normalization in ConvolvedFFTPower' in str(e) :
                # this is a failure mode we sometimes have unfortunately
                if tm.comm.rank == 0 :
                    with open(f'{wrk_dir}/plk_msg_{augment}.info', 'w') as fp :
                        fp.write(f'{e}\n')
                continue
            raise

        if tm.comm.rank == 0 :
            np.savez(f'{wrk_dir}/NEW_plk_{augment}.npz', **plk_result)

        # all done!
        if tm.comm.rank == 0 :
            report_state(running_idx, seed_idx, lightcone_idx, 0 if successful else 2)
