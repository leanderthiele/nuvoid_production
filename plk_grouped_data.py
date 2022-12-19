from glob import glob
import subprocess
import re

import numpy as np

codebase = '/home/lthiele/nuvoid_production'
database = '/scratch/gpfs/lthiele/nuvoid_production'

def get_setting_from_info(fname, name) :
    with open(fname, 'r') as f :
        for line in f :
            line = line.strip().split('=')
            if len(line) != 2 :
                continue
            if line[0] == name :
                return float(line[1])


def get_cosmo(cosmo_idx, cache={}) :
    keys = ['omegabh', 'omegach2', 'theta', 'logA', 'ns', 'Mnu']
    if cosmo_idx not in cache :
        result = subprocess.run(f'{codebase}/sample_prior {cosmo_idx} '\
                                f'5 {codebase}/mu_cov_plikHM_TTTEEE_lowl_lowE.dat '\
                                f'1 {codebase}/mnu_prior.dat',
                                shell=True, capture_output=True, check=True)
        cache[cosmo_idx] = dict(zip(keys, map(float, result.stdout.split(b','))))
        print(f'{cosmo_idx} ns={cache[cosmo_idx]["ns"]}')
    return cache[cosmo_idx]


def get_hod(cosmo_idx, hod_hash) :
    hod_file = f'{database}/cosmo_varied_{cosmo_idx}/lightcones/{hod_hash}/hod.info'
    keys = ['hod_transfP1', 'hod_abias', 'hod_log_Mmin', 'hod_sigma_logM',
            'hod_log_M0', 'hod_log_M1', 'hod_alpha', 'hod_transf_eta_cen',
            'hod_transf_eta_sat', 'hod_mu_Mmin', 'hod_mu_M1', ]
    return {k: get_setting_from_info(hod_file, k) for k in keys}

def split_path(path) :

    m1 = re.search('(?<=cosmo_varied_)[0-9]*', path)
    assert m1 is not None
    cosmo_idx = int(m1[0])

    m2 = re.search('(?<=lightcones/)[a-f,0-9]*', path)   
    assert m2 is not None
    hod_hash = m2[0]
    assert len(hod_hash) == 32

    return cosmo_idx, hod_hash

param_names = None
params = []
cosmo_indices = []
k = None
p0k = []
p2k = []
p4k = []

hod_idx = 1
while True :
    result = subprocess.run(f'{codebase}/mysql_driver_NEW get_run {hod_idx}',
                            shell=True, check=True, capture_output=True)

    # increment before we forget it!
    hod_idx += 1
    if hod_idx % 100 == 0 :
        print(f'At hod_idx={hod_idx}')

    # these are all strings
    cosmo_idx, hod_hash, state, plk_state = result.stdout.decode().split()
    cosmo_idx = int(cosmo_idx)
    if cosmo_idx < 0 :
        # reached end
        break
    if state != 'success' or plk_state != 'success' :
        continue
    wrk_dir = f'{database}/cosmo_varied_{cosmo_idx}/lightcones/{hod_hash}'
    fnames = glob(f'{wrk_dir}/NEW_plk_[0-9]*.npz')
    if len(fnames) != 8 :
        continue

    cosmo = get_cosmo(cosmo_idx)
    hod = get_hod(cosmo_idx, hod_hash)
    param_names_ = list(cosmo.keys()) + list(hod.keys())
    if param_names is None :
        param_names = param_names_
    else :
        assert param_names == param_names_
    all_successful = True
    this_p0k = np.empty((8, 39))
    this_p2k = np.empty((8, 39))
    this_p4k = np.empty((8, 39))
    for ii, fname in enumerate(fnames) :
        try :
            with np.load(fname) as fp :
                if k is None :
                    k = fp['k']
                else :
                    assert np.allclose(k, fp['k'])
                this_p0k[ii] = fp['p0k']
                this_p2k[ii] = fp['p2k']
                this_p4k[ii] = fp['p4k']
        except ValueError :
            print(f'Problem with file {fname}')
            all_successful = False
            break
    if all_successful :
        params.append(list(cosmo.values()) + list(hod.values()))
        cosmo_indices.append(cosmo_idx)
        p0k.append(this_p0k)
        p2k.append(this_p2k)
        p4k.append(this_p4k)
        
np.savez('plk_grouped_data.npz',
         param_names=param_names,
         cosmo_indices=np.array(cosmo_indices, dtype=int),
         params=np.array(params, dtype=np.float32),
         k=k,
         p0k=np.array(p0k, dtype=np.float32),
         p2k=np.array(p2k, dtype=np.float32),
         p4k=np.array(p4k, dtype=np.float32))
