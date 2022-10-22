""" Convert from theta_MC to H0 and do the necessary conversions
to our parameterization.
This is useful because it approximately decouples the CMB prior
from any neutrino mass sum we want to set.

Command line arguments:
[1...6] 5-parameter LCDM, in the order:
        omegabh2, omegach2, theta, logA, ns
[7] sum of neutrino masses in eV

Prints out comma separated list of:
    Omega_B, Omega_M, h0, A_s, n_s, M_nu
which we can then use to run our code
(which is in this parameterization)
"""

from sys import argv

import camb
import math

omegabh2, omegach2, thetaMC, logA, ns, Mnu = map(float, argv[1:])

# we want to keep Omega_M fixed, which is why we need to change omegach2
# depending on neutrino mass sum

p = camb.model.CAMBparams()
p.set_cosmology(H0=None, cosmomc_theta=thetaMC*1e-2,
                ombh2=omegabh2, omch2=omegach2, omk=0.0,
                neutrino_hierarchy='degenerate', num_massive_neutrinos=3, mnu=Mnu)

Omega_B = p.omegab
Omega_M = p.omegam
h0 = p.H0 * 1e-2
A_s = 1e-10 * math.exp(logA)
n_s = ns
M_nu = Mnu

print(f'{Omega_B:.8f},{Omega_M:.8f},{h0:.8f},{A_s:.8e},{n_s:.8f},{M_nu:.8f}')
