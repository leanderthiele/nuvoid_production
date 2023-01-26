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
from vg_plk import PLKCalc

# version of the fiducial HOD
VERSION = int(argv[1])
TABLE = argv[2]
assert TABLE in ['fiducials', 'derivatives', ]

comm_world = MPI.COMM_WORLD
world_rank = comm_world.Get_rank()

codebase = '/home/lthiele/nuvoid_production'
CPUS_PER_TASK = int(os.environ['CPUS_PER_TASK'])

# get rid of annoying warnings from nbodykit
for w in [np.ComplexWarning, np.VisibleDeprecationWarning, ] :
    warnings.filterwarnings('ignore', category=w)

def report_state(running_idx, sim_idx, lightcone_idx, state) :
    subprocess.run(f'{codebase}/mysql_driver end_{TABLE}_vgplk {VERSION} {running_idx} {sim_idx} {lightcone_idx} {state}',
                   shell=True, check=True)

with NBL.TaskManager(cpus_per_task=CPUS_PER_TASK, use_all_cpus=True) as tm :
    
    plk_calc = PLKCalc(comm=tm.comm)

    # we make a fake long list
    for ii in tm.iterate(list(range(10000))) :
        
        # get the sample we should work on
        if tm.comm.rank == 0 :

            result = subprocess.run(f'{codebase}/mysql_driver create_{TABLE}_vgplk {VERSION}',
                                    shell=True, capture_output=True, check=True)

            # these are strings!
            running_idx, sim_idx, lightcone_idx, hod_hash = result.stdout.strip().decode().split()
            if int(sim_idx) < 0 :
                # no work left
                continue

            subprocess.run(f'{codebase}/mysql_driver start_{TABLE}_vgplk {VERSION} {running_idx} {sim_idx} {lightcone_idx}',
                           shell=True, check=True)

            # this is where we operate
            wrk_dir = f'/scratch/gpfs/lthiele/nuvoid_production/cosmo_{"fiducial" if TABLE=="fiducials" else "varied"}_{sim_idx}/{"lightcones" if TABLE=="fiducials" else "derivatives"}/{hod_hash}'

            lightcone_file = f'{wrk_dir}/lightcone_{lightcone_idx}.bin'
            voids_file = f'{wrk_dir}/voids_{lightcone_idx}/sky_positions_central_{lightcone_idx}.out'

            if not os.path.isfile(lightcone_file) :
                print(f'{lightcone_file} not found!', file=sys.stderr)
                report_state(running_idx, sim_idx, lightcone_idx, 1)
                continue
            if not os.path.isfile(voids_file) :
                print(f'{voids_file} not found!', file=sys.stderr)
                report_state(running_idx, sim_idx, lightcone_idx, 1)
                continue
            with open(voids_file, 'r') as f :
                first_line = f.readline()
                if first_line[0] != '#' :
                    print(f'{voids_file} corrupted!', file=sys.stderr)
                    report_state(running_idx, sim_idx, lightcone_idx, 1)
                    continue
        else :
            lightcone_file = None
            voids_file = None

        # let everyone on the team know which files we should operate on
        lightcone_file = tm.comm.bcast(lightcone_file)
        voids_file = tm.comm.bcast(voids_file)

        augment = re.search('(?<=lightcone_)[0-9]*', lightcone_file)[0]

        # consistency check
        augment1 = re.search('(?<=voids_)[0-9]*', voids_file)[0]
        augment2 = re.search('(?<=central_)[0-9]*', voids_file)[0]
        assert augment == augment1 and augment == augment2

        successful = False
        try :
            plk_result = plk_calc.compute_from_fnames(lightcone_file, voids_file)
            successful = True
        except ValueError as e :
            if 'normalization in ConvolvedFFTPower' in str(e) :
                # this is a failure mode we sometimes have unfortunately
                if tm.comm.rank == 0 :
                    with open(f'{wrk_dir}/vgplk_msg_{augment}.info', 'w') as fp :
                        fp.write(f'{e}\n')
                continue
            raise

        if tm.comm.rank == 0 :
            np.savez(f'{wrk_dir}/NEW_vgplk_{augment}.npz', **plk_result)

        # all done!
        if tm.comm.rank == 0 :
            report_state(running_idx, sim_idx, lightcone_idx, 0 if successful else 2)
