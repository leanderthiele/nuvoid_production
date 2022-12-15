import os
import sys
import re
from glob import glob
import subprocess
import numpy as np

import nbodykit.lab as NBL

from plk import PLKCalc

codebase = '/home/lthiele/nuvoid_production'
CPUS_PER_TASK = int(os.environ['CPUS_PER_TASK'])

def report_state(cosmo_idx, hod_idx, state) :
    subprocess.run(f'{codebase}/mysql_driver end_plk {cosmo_idx} {hod_idx} {state}',
                   shell=True, check=True)

with NBL.TaskManager(cpus_per_task=CPUS_PER_TASK) as tm :
    
    plk_calc = PLKCalc()

    # we make a fake long list
    for ii in tm.iterate(list(range(10000))) :
        
        # get the sample we should work on
        result = subprocess.run(f'{codebase}/mysql_driver create_plk',
                                shell=True, capture_output=True, check=True)

        # these are strings!
        cosmo_idx, hod_idx, hod_hash = result.stdout.strip().decode().split()
        if int(cosmo_idx) < 0 :
            # no work left
            continue

        subprocess.run(f'{codebase}/mysql_driver start_plk {cosmo_idx} {hod_idx}',
                       shell=True, check=True)

        # this is where we operate
        wrk_dir = f'/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_{cosmo_idx}/lightcones/{hod_hash}'

        lightcone_files = glob(f'{wrk_dir}/lightcone_[0-9]*.bin')
        if not lightcone_files :
            print(f'No lightcones found in {wrk_dir}!', file=sys.stderr)
            report_state(cosmo_idx, hod_idx, 1)
            continue

        # iterate over the lightcones
        for lightcone_file in lightcone_files :
            
            augment = re.search('(?<=lightcone_)[0-9]*', lightcone_file)
            plk_result = plk_calc.compute_from_fname(lightcone_file)
            np.savez(f'{wrk_dir}/plk_{augment}.npz', **plk_result)

        # all done!
        report_state(cosmo_idx, hod_idx, 0)
