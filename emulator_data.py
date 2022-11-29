from glob import glob
import subprocess
import re

import numpy as np

from tqdm import tqdm

vide_out = 'untrimmed_dencut'

codebase = '/home/lthiele/nuvoid_production'
database = '/scratch/gpfs/lthiele/nuvoid_production'

RMIN = 30.0
RMAX = 80.0
NBINS = 32
ZEDGES = [0.53, ] # can be empty

def get_setting_from_info(fname, name) :
    with open(fname, 'r') as f :
        for line in f :
            line = line.strip().split('=')
            if len(line) != 2 :
                continue
            if line[0] == name :
                return float(line[1])


def get_cosmo(cosmo_idx, cache={}) :
    keys = ['Omega_m', 'Omega_b', 'h', 'n_s', 'A_s', 'M_nu', ]
    if cosmo_idx not in cache :
        cosmo_file = f'{database}/cosmo_varied_{cosmo_idx}/cosmo.info'
        cache[cosmo_idx] = {k: get_setting_from_info(cosmo_file, k) for k in keys}
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


def get_hist(void_file, cache={}) :
    # returns a string!
    if not cache :
        cache['Rbins'] = np.linspace(RMIN, RMAX, NBINS+1)
        zedges = sorted(z for z in ZEDGES)
        zedges.insert(0, 0.0)
        zedges.append(100.0)
        cache['zbins'] = np.array(zedges)
    try :
        R, z = np.loadtxt(void_file, usecols=(4,5,), unpack=True)
    except ValueError :
        print(f'Error for: {void_file}')
        return None, None
    h, _, _ = np.histogram2d(z, R, bins=(cache['zbins'], cache['Rbins'], ))
    h = h.flatten().astype(int)
    out = ' '.join(map(str, h))
    return out, h


def get_loglike(void_file, cache={}) :
    if not cache :
        cache['cmass_hist'], _ = get_hist(f'/tigress/lthiele/boss_dr12/voids/sample_test/'\
                                          f'{vide_out}_centers_central_test.out')
        assert cache['cmass_hist'] is not None
        cache['total_bins'] = NBINS * ( len(ZEDGES) + 1 )
    sim_hist_str, sim_hist_arr = get_hist(void_file)
    if sim_hist_str is None :
        return None, None
    result = subprocess.run(f'{codebase}/vsf_like {cache["total_bins"]} {cache["cmass_hist"]} {sim_hist_str}',
                            shell=True, capture_output=True, check=True)
    return float(result.stdout), sim_hist_arr


# do a glob for all available void catalogs
print('globbing...')
void_files = glob(f'{database}/cosmo_varied_*/emulator/*/sample_*/{vide_out}_centers_central_*.out')
print(f'Found {len(void_files)} void catalogs.')

param_names = None
params = []
values = []
hists  = []

for f in tqdm(void_files) :
    cosmo_idx, hod_hash = split_path(f)
    cosmo = get_cosmo(cosmo_idx)
    hod = get_hod(cosmo_idx, hod_hash)
    param_names_ = list(cosmo.keys()) + list(hod.keys())
    if param_names is None :
        param_names = param_names_
    else :
        assert param_names == param_names_
    L, h = get_loglike(f)
    if L is None :
        continue
    params.append(list(cosmo.values()) + list(hod.values()))
    hists.append(h)
    values.append(L)

np.savez(f'emulator_data_RMIN{RMIN}_RMAX{RMAX}_NBINS{NBINS}_ZEDGES{"-".join(map(str, ZEDGES))}_{vide_out}.npz',
         param_names=param_names,
         params=np.array(params),
         values=np.array(values),
         hists=np.array(hists))
