""" Assemble all vgplk into a convenient file

Output file contents:
    param_names [Nparams]
    params [Nsamples, Nparams]
    cosmo_indices [Nsamples]
    Rmin[Nr]
    k[Nk]
    vgplk[Nparams, 8, Nr, [ell=0,2], Nk]
        entries are NaN if not enough files available
    Nvoids[Nparams, 8, Nr] -- how many voids were used for the computation
"""

import sys
from glob import glob
import re
import subprocess

import numpy as np

from tqdm import tqdm

codebase = '/home/lthiele/nuvoid_production'
database = '/scratch/gpfs/lthiele/nuvoid_production'

# hardcoded...
RMAX = 80

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
Rmin = None
k = None
vgplk = []
Nvoids = []

def save() :
    np.savez('collected_vgplk.npz',
             param_names=param_names,
             cosmo_indices=np.array(cosmo_indices, dtype=int),
             params=np.array(params, dtype=np.float32),
             Rmin=Rmin, k=k,
             vgplk=np.array(vgplk, dtype=np.float32),
             Nvoids=np.array(Nvoids, dtype=np.float32))

for cosmo_idx in tqdm(range(130)) :
    
    hod_dirs = glob(f'{database}/cosmo_varied_{cosmo_idx}/lightcones/[a-f,0-9]*')
    if not hod_dirs :
        continue
    for hod_dir in hod_dirs :

        if (len(vgplk)+1) % 100 == 0 :
            print(f'Have collected {len(vgplk)} samples')
        if (len(vgplk)+1) % 1000 == 0 :
            save()

        vgplk_files = glob(f'{hod_dir}/NEW_vgplk_[0-9]*.npz')
        if not vgplk_files :
            continue
        one_successful = False
        this_vgplk = np.full((8, 3, 2, 40), float('nan'))
        this_Nvoids = np.full((8, 3), float('nan'))
        for ii, vgplk_file in enumerate(vgplk_files) :
            try :
                with np.load(vgplk_file) as f :
                    k_ = f['k']
                    if k is None :
                        k = k_
                        assert len(k) == this_vgplk.shape[-1]
                    else :
                        assert np.allclose(k, k_)
                    Rmin_ = np.array(sorted(list(map(lambda s: int(s.split('Rmin')[1]),
                                                 filter(lambda s: 'p0k' in s, list(f.keys()))))))
                    if Rmin is None :
                        Rmin = Rmin_
                        assert len(Rmin) == this_vgplk.shape[1]
                    else :
                        assert np.allclose(Rmin, Rmin_)
                    for jj, Rmin_ in enumerate(Rmin) :
                        for kk, ell in enumerate([0, 2]) :
                            this_vgplk[ii, jj, kk, :] = f[f'p{ell}k_Rmin{Rmin_}']
                    one_successful = True
            except ValueError :
                print(f'Problem with {vgplk_file}')
                continue
            augment = int(re.search('(?<=NEW_vgplk_)[0-9*]', vgplk_file)[0])
            voids_file = f'{hod_dir}/voids_{augment}/sky_positions_central_{augment}.out'
            R = np.loadtxt(voids_file, usecols=(3,))
            for jj, r in enumerate(Rmin) :
                this_Nvoids[ii, jj] = np.count_nonzero((R<RMAX)*(R>r))
        if one_successful : 
            hod_hash = hod_dir.rstrip('/').split('/')[-1]
            cosmo = get_cosmo(cosmo_idx)
            hod = get_hod(cosmo_idx, hod_hash)
            param_names_ = list(cosmo.keys()) + list(hod.keys())
            if param_names is None :
                param_names = param_names_
            else :
                assert param_names == param_names_
            params.append(list(cosmo.values()) + list(hod.values()))
            cosmo_indices.append(cosmo_idx)
            vgplk.append(this_vgplk)
            Nvoids.append(this_Nvoids)

# final save
save()
