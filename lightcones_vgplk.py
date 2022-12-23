import os
import os.path
import sys
import re
from glob import glob
import subprocess
import warnings

import numpy as np

import nbodykit.lab as NBL

from mpi4py import MPI
from vg_plk import PLKCalc

comm_world = MPI.COMM_WORLD
world_rank = comm_world.Get_rank()

codebase = '/home/lthiele/nuvoid_production'
CPUS_PER_TASK = int(os.environ['CPUS_PER_TASK'])

# get rid of annoying warnings from nbodykit
for w in [np.ComplexWarning, np.VisibleDeprecationWarning, ] :
    warnings.filterwarnings('ignore', category=w)

def report_state(cosmo_idx, hod_idx, state) :
    subprocess.run(f'{codebase}/mysql_driver end_vgplk {cosmo_idx} {hod_idx} {state}',
                   shell=True, check=True)

with NBL.TaskManager(cpus_per_task=CPUS_PER_TASK, use_all_cpus=True) as tm :
    
    plk_calc = PLKCalc(comm=tm.comm)

    # we make a fake long list
    for ii in tm.iterate(list(range(10000))) :
        
        # get the sample we should work on
        if tm.comm.rank == 0 :

            result = subprocess.run(f'{codebase}/mysql_driver create_vgplk',
                                    shell=True, capture_output=True, check=True)

            # these are strings!
            cosmo_idx, hod_idx, hod_hash = result.stdout.strip().decode().split()
            if int(cosmo_idx) < 0 :
                # no work left
                continue

            subprocess.run(f'{codebase}/mysql_driver start_vgplk {cosmo_idx} {hod_idx}',
                           shell=True, check=True)

            # this is where we operate
            wrk_dir = f'/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_{cosmo_idx}/lightcones/{hod_hash}'

            lightcone_files_ = glob(f'{wrk_dir}/lightcone_[0-9]*.bin')

            lightcone_files = []
            voids_files = []
            for lightcone_file in lightcone_files_ :
                augment = re.search('(?<=lightcone_)[0-9]*', lightcone_file)[0]
                voids_file = f'{wrk_dir}/voids_{augment}/sky_positions_central_{augment}.out'
                if not os.path(voids_file) :
                    continue
                with open(voids_file, 'r') as f :
                    first_line = f.readline()
                    if first_line[0] != '#' :
                        # corrupted file
                        continue
                # all good, we can use this one
                lightcone_files.append(lightcone_file)
                voids_files.append(voids_file)

            if not lightcone_files :
                print(f'No lightcones found in {wrk_dir}!', file=sys.stderr)
                report_state(cosmo_idx, hod_idx, 1)
                continue
        else :
            lightcone_files = None
            voids_files = None

        # let everyone on the team know which files we should operate on
        lightcone_files = tm.comm.bcast(lightcone_files)
        voids_files = tm.comm.bcast(voids_files)

        # iterate over the lightcones
        at_least_one_successful = False
        for lightcone_file, voids_file in zip(lightcone_files, voids_files) :
            
            augment = re.search('(?<=lightcone_)[0-9]*', lightcone_file)[0]

            # consistency check
            augment1 = re.search('(?<=voids_)[0-9]*', voids_file)[0]
            augment2 = re.search('(?<=central_)[0-9*]', voids_file)[0]
            assert augment == augment1 and augment == augment2

            try :
                plk_result = plk_calc.compute_from_fnames(lightcone_file, voids_file)
            except ValueError as e :
                if 'normalization in ConvolvedFFTPower' in str(e) :
                    # this is a failure mode we sometimes have unfortunately
                    if tm.comm.rank == 0 :
                        with open(f'{wrk_dir}/vgplk_msg_{augment}.info', 'w') as fp :
                            fp.write(f'{e}\n')
                    continue
                raise

            at_least_one_successful = True
            if tm.comm.rank == 0 :
                np.savez(f'{wrk_dir}/vgplk_{augment}.npz', **plk_result)

        # all done!
        if tm.comm.rank == 0 :
            report_state(cosmo_idx, hod_idx, 0 if at_least_one_successful else 2)
