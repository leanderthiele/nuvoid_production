""" The ultimate script to assemble all data vectors
Command line arguments:
    [1] case (trials, fiducials_v%d, derivatives_v%d)
"""

import os.path
import sys
from sys import argv
import subprocess
import re

import numpy as np
from tqdm import tqdm

from datavector import VSF_ZEDGES, VSF_REDGES, VGPLK_R, VGPLK_ELL, VGPLK_K, PLK_ELL, PLK_K

codebase = '/home/lthiele/nuvoid_production'
database = '/scratch/gpfs/lthiele/nuvoid_production'

# this is where outputs go
filebase = '/tigress/lthiele/nuvoid_production'

# the output data
sim_idx = []
hod_hi_word = []
hod_lo_word = []
lc_idx = []
data = []
# only if not fiducial
param_names = None
params = []

def get_setting_from_info(fname, name) :
    with open(fname, 'r') as f :
        for line in f :
            line = line.strip().split('=')
            if len(line) != 2 :
                continue
            if line[0] == name :
                return float(line[1])


def get_cosmo(cosmo_idx, cache={}) :
    if cosmo_idx not in cache :
        result = subprocess.run(f'{codebase}/mysql_driver get_cosmo {cosmo_idx}',
                                shell=True, capture_output=True, check=True)
        lines = result.stdout.decode().split()
        out = {}
        for line in lines :
            k, v = line.split('=')
            out[k] = float(v)
        cache[cosmo_idx] = out
    return cache[cosmo_idx]

def get_hod(path) :
    hod_file = f'{path}/hod.info'
    keys = ['hod_transfP1', 'hod_abias', 'hod_log_Mmin', 'hod_sigma_logM',
            'hod_log_M0', 'hod_log_M1', 'hod_alpha', 'hod_transf_eta_cen',
            'hod_transf_eta_sat', 'hod_mu_Mmin', 'hod_mu_M1', ]
    return {k: get_setting_from_info(hod_file, k) for k in keys}

def get_datavec_from_fnames (voids_fname, vgplk_fname, plk_fname) :
    if not os.path.isfile(voids_fname) :
        raise FileNotFoundError(f'no voids file: {voids_fname}')
    with open(voids_fname, 'r') as f :
        first_line = f.readline()
        if first_line[0] != '#' : # indicates corrupted file
            raise ValueError('corrupted voids file')

    if not os.path.isfile(vgplk_fname) :
        raise FileNotFoundError(f'no vgplk file: {vgplk_fname}')
    try :
        fvgplk = np.load(vgplk_fname)
    except ValueError : # indicates corrupted file
        raise ValueError('corrupted vgplk file')
    if not np.allclose(VGPLK_K, fvgplk['k']) :
        raise ValueError('vgplk k does not match')

    if not os.path.isfile(plk_fname) :
        raise FileNotFoundError(f'no plk file: {plk_fname}')
    try :
        fplk = np.load(plk_fname)
    except ValueError : # indicates corrupted file
        raise ValueError('corrupted plk file')
    if not np.allclose(PLK_K, fplk['k']) :
        raise ValueError('plk k does not match')

    # compute VSF
    z, R = np.loadtxt(voids_fname, usecols=(2,3), unpack=True)
    out = np.histogram2d(z, R, bins=[VSF_ZEDGES, VSF_REDGES])[0].flatten().astype(float)

    # append VGPLK
    out = np.concatenate([out, *[fvgplk[f'p{ell}k_Rmin{R}'] for R in VGPLK_R for ell in VGPLK_ELL]])
    fvgplk.close()

    # append PLK
    out = np.concatenate([out, *[fplk[f'p{ell}k'] for ell in PLK_ELL]])
    fplk.close()

    return out.astype(np.float32)

def get_datavec (path, lc) :
    """ return the data vector in path for lightcone lc,
        raises FileNotFoundError if any of the required files
        does not exist or is corrupted.
    """

    voids_fname = f'{path}/voids_{lc}/sky_positions_central_{lc}.out'
    vgplk_fname = f'{path}/NEW_vgplk_{lc}.npz'
    plk_fname = f'{path}/NEW_plk_{lc}.npz'

    return get_datavec_from_fnames(voids_fname, vgplk_fname, plk_fname)


def handle_dir (d, case, version) :
    
    global sim_idx
    global hod_hi_word
    global hod_lo_word
    global lc_idx
    global data
    global param_names
    global params

    path = f'{database}/{d}'

    hod_hash = re.search('[a-f,0-9]{32}', path)[0]
    if case != 'fiducials' :
        cosmo_idx = int(re.search('(?<=cosmo_varied_)[0-9]*', path)[0])
        cosmo = get_cosmo(cosmo_idx)
        hod = get_hod(path)
        param_names_ = list(cosmo.keys()) + list(hod.keys())
        if param_names is None :
            param_names = param_names_
        else :
            assert param_names == param_names_
        this_params = np.array(list(cosmo.values()) + list(hod.values()))
        this_sim_idx = cosmo_idx
    else :
        this_params = None
        this_sim_idx = int(re.search('(?<=cosmo_fiducial_)[0-9]*', path)[0])

    for this_lc_idx in range(96) :
        try :
            this_data = get_datavec(path, this_lc_idx)
        except Exception as e :
            if case != 'trials' or not isinstance(e, FileNotFoundError) :
                # FileNotFound is expected as we don't have all the lightcones
                with open(f'{filebase}/errors_{case}{"" if version is None else f"_v{version}"}.log', 'a') as f :
                    f.write(f'{path}::{this_lc_idx}\t{e}\n')
            continue
        sim_idx.append(this_sim_idx)
        hod_hi_word.append(int(hod_hash[:16], base=16))
        hod_lo_word.append(int(hod_hash[16:], base=16))
        lc_idx.append(this_lc_idx)
        data.append(this_data)
        if this_params is not None :
            params.append(this_params)

def save (fname) :
    out = {'sim_idx': np.array(sim_idx, dtype=np.uint16),
           'hod_hi_word': np.array(hod_hi_word, dtype=np.uint64),
           'hod_lo_word': np.array(hod_lo_word, dtype=np.uint64),
           'lc_idx': np.array(lc_idx, dtype=np.uint8),
           'data': np.array(data, dtype=np.float32)
          }
    if params :
        assert param_names is not None
        out['param_names'] = param_names
        out['params'] = np.array(params, dtype=np.float32)
    np.savez(fname, **out)

if __name__ == '__main__' :
    
    case = argv[1]
    outfile = f'{filebase}/datavectors_{case}.npz'

    if case == 'trials' :
        version = None
    else :
        version = int(case.split('_v')[1])
        case = case.split('_v')[0]
    cmd = f'{codebase}/mysql_driver get_successful_{case}'
    if version is not None :
        cmd = f'{cmd} {version}'

    dirs = list(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split())
    print(f'Will work on {len(dirs)} directories')
    for ii, d in enumerate(tqdm(dirs)) :
        handle_dir(d, case, version)
        if (ii+1) % 100 == 0 :
            save(outfile)
    
    # final save
    save(outfile)
