""" This is for random catalogs, so the only purpose is correct
accounting for the mask (which couples with void radius).

Output file contents:
    param_names [Nparams]
    params [Nsamples, Nparams]
    cosmo_indices [Nsamples]
    Nvoids [Nsamples, 8] -- set to 0 if None found our files corrupted
    RA, DEC, Z, R
"""

import sys
import os.path
from glob import glob
import subprocess
import re

import numpy as np

RMIN = 30

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
Nvoids = np.empty((0, 8), dtype=int)
RA = np.empty(0, dtype=np.float32)
DEC = np.empty(0, dtype=np.float32)
Z = np.empty(0, dtype=np.float32)
R = np.empty(0, dtype=np.float32)

def save() :
    np.savez('collected_voids.npz',
             param_names=param_names,
             cosmo_indices=np.array(cosmo_indices, dtype=int),
             params=np.array(params, dtype=np.float32),
             Nvoids=Nvoids, RA=RA, DEC=DEC, Z=Z, R=R)

hod_idx = 1
while True :
    result = subprocess.run(f'{codebase}/mysql_driver get_run {hod_idx}',
                            shell=True, check=True, capture_output=True)

    hod_idx += 1
    if hod_idx % 100 == 0 :
        print(f'At hod_idx={hod_idx}')

    if hod_idx % 1000 == 0 :
        save()

    # these are all strings
    cosmo_idx, hod_hash, state, plk_state, voids_state = result.stdout.decode().split()
    cosmo_idx = int(cosmo_idx)
    if cosmo_idx < 0 :
        # reached end
        break
    if state != 'success' or voids_state != 'success' :
        continue

    wrk_dir = f'{database}/cosmo_varied_{cosmo_idx}/lightcones/{hod_hash}'
    voids_dirs = glob(f'{wrk_dir}/voids_[0-9]*')
    if not voids_dirs :
        continue

    cosmo = get_cosmo(cosmo_idx)
    hod = get_hod(cosmo_idx, hod_hash)
    param_names_ = list(cosmo.keys()) + list(hod.keys())
    if param_names is None :
        param_names = param_names_
    else :
        assert param_names == param_names_

    this_Nvoids = np.zeros(8, dtype=int)
    for ii, voids_dir in enumerate(voids_dirs) :
        # find the augmentation index
        m = re.search('(?<=voids_)[0-9]*', voids_dir)
        assert m is not None
        augment = int(m[0])

        sky_pos_fname = f'{voids_dir}/sky_positions_central_{augment}.out'
        if not os.path.isfile(sky_pos_fname) :
            print(f'Could not find all files in {voids_dir}', file=sys.stderr)
            continue
        with open(sky_pos_fname, 'r') as f :
            first_line = f.readline()
            if first_line[0] != '#' :
                print(f'Corrupted file {sky_pos_fname}', file=sys.stderr)
                continue
        ra, dec, z, r = np.loadtxt(sky_pos_fname, usecols=(0,1,2,3), unpack=True, dtype=np.float32)
        select = r > RMIN
        if np.count_nonzero(select) == 0 :
            continue
        ra = ra[select]
        dec = dec[select]
        z = z[select]
        r = r[select]
        this_Nvoids[ii] = len(ra)
        RA = np.concatenate((RA, ra))
        DEC = np.concatenate((DEC, dec))
        Z = np.concatenate((Z, z))
        R = np.concatenate((R, r))

    if np.count_nonzero(this_Nvoids) == 0 :
        continue

    params.append(list(cosmo.values()) + list(hod.values()))
    cosmo_indices.append(cosmo_idx)
    Nvoids = np.concatenate((Nvoids, this_Nvoids.reshape(1, -1)), axis=0)
        


# final output
save()
