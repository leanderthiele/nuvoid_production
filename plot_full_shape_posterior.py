from sys import argv
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

import corner

root = argv[1]

discard = 0 # burn in

param_names_files = glob(f'{root}/*.paramnames')
assert len(param_names_files) == 1
param_names_file = param_names_files[0]
param_names = []
with open(param_names_file, 'r') as f :
    for line in f :
        param_names.append(f'${line.split(" ")[-1]}$')
print(param_names)
dim = len(param_names)

txt_files = glob(f'{root}/*.txt'

x = np.empty((0, dim))
for txt_file in txt_files :
    a = np.loadtxt(txt_file)
    n = a[0]
    neglkl = a[1] # unused
    x_ = a[2:]
    x_ = np.repeat(x_, n.astype(int), axis=0)
    x_ = x_[:, discard:]
    x = np.concatenate((x, x_), axis=0)

fig, ax = plt.subplots(ncols=dim, nrows=dim, figsize=(20,20))
corner.corner(x, labels=param_names, plot_datapoints=False, fig=fig)

fig.savefig(f'{root}/posterior_corner.pdf', bbox_inches='tight')
