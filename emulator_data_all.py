from glob import glob
import subprocess
import re

import numpy as np

from tqdm import tqdm

vide_out = 'untrimmed_dencut'

codebase = '/home/lthiele/nuvoid_production'
database = '/scratch/gpfs/lthiele/nuvoid_production'

MAX_COUNT = 2000 # maximum number of voids supported
RMIN = 30

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
    hod_file = f'{database}/cosmo_varied_{cosmo_idx}/emulator/{hod_hash}/hod.info'
    keys = ['hod_transfP1', 'hod_abias', 'hod_log_Mmin', 'hod_sigma_logM',
            'hod_log_M0', 'hod_log_M1', 'hod_alpha', 'hod_transf_eta_cen',
            'hod_transf_eta_sat', 'hod_mu_Mmin', 'hod_mu_M1', ]
    return {k: get_setting_from_info(hod_file, k) for k in keys}

def split_path(path) :

    m1 = re.search('(?<=cosmo_varied_)[0-9]*', path)
    assert m1 is not None
    cosmo_idx = int(m1[0])

    m2 = re.search('(?<=emulator/)[a-f,0-9]*' , path)   
    assert m2 is not None
    hod_hash = m2[0]
    assert len(hod_hash) == 32

    return cosmo_idx, hod_hash

# do a glob for all available void catalogs
print('globbing...')
void_files = glob(f'{database}/cosmo_varied_*/emulator/*/sample_*/{vide_out}_centers_central_*.out')
print(f'Found {len(void_files)} void catalogs.')

param_names = None
params = []
cosmo_indices = []
radii  = []
redshifts = []
density_constrasts = []
num_parts = []
ellips = []
# not sure what these are...
eig1s = []
eig2s = []
eig3s = []

for f in tqdm(void_files) :
    cosmo_idx, hod_hash = split_path(f)
    cosmo = get_cosmo(cosmo_idx)
    hod = get_hod(cosmo_idx, hod_hash)
    param_names_ = list(cosmo.keys()) + list(hod.keys())
    if param_names is None :
        param_names = param_names_
    else :
        assert param_names == param_names_
    with open(f, 'r') as fp :
        first_line = fp.readline()
        if first_line[0] != '#' :
            print(f'Corrupted file {f} (no header)')
            continue
    try :
        R, z, density_contrast, num_part = np.loadtxt(f, usecols=(4,5,8,9), unpack=True)
        ellip, eig1, eig2, eig3 = np.loadtxt(f.replace('centers', 'shapes'), usecols=(1,2,3,4), unpack=True)
    except ValueError :
        continue
    if not len(R) == len(ellip) :
        print(f'Corrupted file {f} (centers and shape do not have same length)')
        continue
    select = R > RMIN
    R = R[select]
    z = z[select]
    density_contrast = density_contrast[select]
    num_part = num_part[select]
    ellip = ellip[select]
    eig1 = eig1[select]
    eig2 = eig2[select]
    eig3 = eig3[select]
    R = np.concatenate([R, np.full(MAX_COUNT-len(R), -1)]).astype(np.float32)
    z = np.concatenate([z, np.full(MAX_COUNT-len(z), -1)]).astype(np.float32)
    density_contrast = np.concatenate([density_contrast, np.full(MAX_COUNT-len(density_contrast), -1)]).astype(np.float32)
    num_part = np.concatenate([num_part, np.full(MAX_COUNT-len(num_part), -1)]).astype(np.float32)
    ellip = np.concatenate([ellip, np.full(MAX_COUNT-len(ellip), -1)]).astype(np.float32)
    eig1 = np.concatenate([eig1, np.full(MAX_COUNT-len(eig1), -1)]).astype(np.float32)
    eig2 = np.concatenate([eig2, np.full(MAX_COUNT-len(eig2), -1)]).astype(np.float32)
    eig3 = np.concatenate([eig3, np.full(MAX_COUNT-len(eig3), -1)]).astype(np.float32)
    radii.append(R)
    redshifts.append(z)
    density_constrasts.append(density_contrast)
    num_parts.append(num_part)
    ellips.append(ellip)
    eig1s.append(eig1)
    eig2s.append(eig2)
    eig3s.append(eig3)
    params.append(list(cosmo.values()) + list(hod.values()))
    cosmo_indices.append(cosmo_idx)

np.savez(f'all_emulator_data_RMIN{RMIN}_{vide_out}.npz',
         param_names=param_names,
         cosmo_indices=np.array(cosmo_indices, dtype=int),
         params=np.array(params, dtype=np.float32),
         radii=np.array(radii, dtype=np.float32),
         redshifts=np.array(redshifts, dtype=np.float32),
         density_contrasts=np.array(density_constrasts, dtype=np.float32),
         num_parts=np.array(num_parts, dtype=np.float32),
         ellips=np.array(ellips, dtype=np.float32),
         eig1s=np.array(eig1s, dtype=np.float32),
         eig2s=np.array(eig2s, dtype=np.float32),
         eig3s=np.array(eig3s, dtype=np.float32))
