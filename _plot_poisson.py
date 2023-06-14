import numpy as np
from matplotlib import pyplot as plt

from _plot_style import *

version = 0
filebase = '/tigress/lthiele/nuvoid_production'
fiducials_fname = f'{filebase}/datavectors_fiducials_v{version}.npz'

with np.load(fiducials_fname) as f :
    data = f['data']

# extract the VSF

fig, ax = plt.subplots(ncols=2, figsize=(5,3))
