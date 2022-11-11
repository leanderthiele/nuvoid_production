from sys import argv

import numpy as np

f = argv[1]
Rmin = float(argv[2])
Rmax = float(argv[3])
Nbins = int(argv[4])
try :
    zedges = sorted(map(float, argv[5].split(',')))
except IndexError :
    zedges = []
# for convenience
zedges.insert(0, 0.0)
zedges.append(100.0)

Redges = np.linspace(Rmin, Rmax, num=Nbins+1)
zedges = np.array(zedges)

R, z = np.loadtxt(f, usecols=(4,5,), unpack=True)
h, _, _ = np.histogram(z, R, bins=(zedges, Redges))
print(','.join(map(str, h.flatten().astype(int))))
