""" Convert from theta_MC to H0.
This is useful because it approximately decouples the CMB prior
from any neutrino mass sum we want to set.

Command line arguments:
[1...6] 6-parameter LCDM, in the order:
        omegabh2, omegach2, theta, tau, logA, ns
[7] sum of neutrino masses in eV
"""

from sys import argv

import camb

omegabh2, omegach2, thetaMC, tau, logA, ns, Mnu = map(float, argv[1:])

# we want to keep Omega_M fixed, which is why we need to change omegach2
# depending on neutrino mass sum

p = camb.model.CAMBparams()
p.set_cosmology(H0=None, cosmomc_theta=thetaMC*1e-2,
                ombh2=omegabh2, omch2=omegach2, omk=0.0,
                neutrino_hierarchy='degenerate', num_massive_neutrinos=3, mnu=Mnu,
                tau=tau)

print(p.H0)
