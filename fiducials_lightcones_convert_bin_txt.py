""" Small helper to convert .bin files to .txt files
Command line arguments:
    [1] input directory
    [2] output directory
    [3] lightcone index
"""

import os.path
from sys import argv

import numpy as np

indir = argv[1]
outdir = argv[2]
lightcone_idx = int(argv[3])

bin_fnames = glob(f'{indir}/lightcone_[0-9]*.bin')
assert(bin_fnames)

bin_fname = f'{indir}/lightcone_{lightcone_idx}.bin'

txt_fname = f'{outdir}/{os.path.splitext(os.path.basename(bin_fname))[0]}.txt'
ra, dec, z = np.fromfile(bin_fname).reshape(3, -1)
v = z * 299792.458
indices = np.arange(len(ra))
content = np.stack([indices, ra, dec, v, ],
                   axis=-1)
np.savetxt(txt_fname, content, fmt='%d 0 0 %.10f %.10f %.8f 1.0000 0.0')
