from sys import argv

import numpy as np

f = argv[1]
Rmin = float(argv[2])
Rmax = float(argv[3])
Nbins = int(argv[4])

col_idx = 4 # corresponding to radius in the vide output

R = np.loadtxt(f, usecols=col_idx)
h, _ = np.histogram(R, bins=Nbins, range=(Rmin, Rmax))
print(','.join(map(str, h)))
