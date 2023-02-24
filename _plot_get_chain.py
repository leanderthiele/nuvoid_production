import re
from glob import glob
import subprocess

import numpy as np

from lfi_load_posterior import load_posterior
from read_txt import read_txt

filebase = '/tigress/lthiele/nuvoid_production'

class ChainContainer :
    def __init__ (self, chain, logprob, param_names, is_fs, stats_str,
                        fid_idx=None, quick_hash=None, version=None, compression_hash=None,
                        model_hash=None, model_settings=None, compression_settings=None,
                        kmax=None, lmax=None) :
        self.chain = chain
        self.logprob = logprob
        self.param_names = param_names
        self.is_fs = is_fs
        self.stats_str = stats_str
        self.fid_idx = fid_idx
        self.quick_hash = quick_hash
        self.version = version
        self.compression_hash = compression_hash
        self.model_hash = model_hash
        self.model_settings = model_settings
        self.compression_settings = compression_settings
        self.kmax = kmax
        self.lmax = lmax


def get_fs (name) :
    
    # burn-in, shouldn't really matter
    discard = 200

    # get these to our naming convention
    map_names = {
                 'omega_b': 'Obh2',
                 'omega_cdm': 'Och2',
                 'h': 'h',
                 'ln10^{10}A_s': 'logA',
                 'n_s': 'ns',
                 'M_tot': 'Mnu',
                 'b^{(1)}_1': 'eft_b1',
                 'b^{(1)}_2': 'eft_b2',
                 'b^{(1)}_{G_2}': 'eft_bG2',
                }

    fsbase = f'{filebase}/{name}'
    param_name_files = glob(f'{fsbase}/*.paramnames')
    assert len(param_name_files) == 1
    param_names = []
    with open(param_name_files[0], 'r') as f :
        param_names.extend(map_names[l.split()[0]] for l in f)

    input_file = f'{fsbase}/log.param'

    # this is a bit hacky but whatever
    class Dummy :
        pass
    data = Dummy()
    data.parameters = {}
    data.cosmo_arguments = {}
    data.path = {}
    full_shape_spectra = Dummy()
    my_planck_prior = Dummy()
    with open(input_file, 'r') as f :
        for line in f :
            exec(line)

    if not hasattr(full_shape_spectra, 'lmax') :
        full_shape_spectra.lmax = 4

    chain = np.empty((0, len(param_names)))
    logprob = np.empty(0)
    txt_files = glob(f'{fsbase}/*.txt')
    for txt_file in txt_files :
        a = np.loadtxt(txt_file)
        repeats = a[:, 0].astype(int)
        logprob = np.concatenate((logprob, np.repeat(a[:, 1], repeats)[discard:]))
        chain = np.concatenate((chain, np.repeat(a[:, 2:], repeats, axis=0)[discard:]), axis=0)

    stats_str = f'$P^{{gg}}_{{{",".join(map(str, range(0, full_shape_spectra.lmax+1, 2)))}}}$'

    return ChainContainer(chain, logprob, param_names, True, stats_str,
                          kmax=full_shape_spectra.kmaxP[0], lmax=full_shape_spectra.lmax)



def get_sbi (fname) :
    
    # for the GPU runs, which are generally shorter, we have optimized the stretch parameter
    # so burn-in is less
    discard = 100 if 'emceegpu' in fname else 1000

    with np.load(f'{filebase}/{fname}') as f :
        chain = f['chain']
        logprob = f['log_prob']
        param_names = list(f['param_names'])
    chain = chain[discard:].reshape(-1, chain.shape[-1])
    logprob = logprob[discard:]

    match = re.search('.*v(\d*).*([a-f,0-9]{32}).*([a-f,0-9]{32}).*', fname)
    version = int(match[1])
    compression_hash = match[2]
    model_hash = match[3]

    # for us during debugging
    quick_hash = f'{compression_hash[:4]}-{model_hash[:4]}'

    if 'fid' in fname :
        match = re.search('.*fid([0-9]*).*', fname)
        fid_idx = int(match[1])
    else :
        fid_idx = None

    model_fname = f'{filebase}/lfi_model_v{version}_{compression_hash}_{model_hash}.sbi'
    compression_fname = f'{filebase}/compression_v{version}_{compression_hash}.dat'

    model_settings, _ = load_posterior(model_fname, None, need_posterior=False)
    compression_settings = read_txt(compression_fname, 'cut_kwargs:', pyobj=True)

    stats_str = '+'.join( \
        map(lambda s: \
            '$N_v$' if s=='vsf' \
            else f'$P^{{vg}}_{{{",".join(map(str, sorted(compression_settings["vgplk_ell"])))}}}$' if s=='vgplk' \
            else f'$P^{{gg}}_{{{",".join(map(str, sorted(compression_settings["plk_ell"])))}}}$' if s=='plk' \
            else s, \
            filter(lambda s: compression_settings[f'use_{s}'], ['vsf', 'vgplk', 'plk'])
           )
        )

    if any(compression_settings[f'use_{s}'] for s in ['vgplk', 'plk', ]) :
        kmax = compression_settings['kmax']
    else :
        kmax = None

    if compression_settings['use_plk'] :
        lmax = max(compression_settings['plk_ell'])
    else :
        lmax = None

    return ChainContainer(chain, logprob, param_names, False, stats_str,
                          fid_idx=fid_idx, quick_hash=quick_hash, version=version,
                          compression_hash=compression_hash, model_hash=model_hash,
                          model_settings=model_settings, compression_settings=compression_settings,
                          kmax=kmax, lmax=lmax)


def get_chain (name, cache={}) :
    # might be called repeatedly for different plots, so cache the results

    if name not in cache :
        if name.startswith('full_shape') :
            cache[name] = get_fs(name)
        else :
            cache[name] = get_sbi(name)

    return cache[name]
