# Import this to get a uniform style

from matplotlib import pyplot as plt
from itertools import cycle

plt.style.use('dark_background')

def savefig (fig, name) :
    fmt = 'png'
    kwargs = dict(bbox_inches='tight', transparent=False)
    outdir = '.'

    fig.savefig(f'{outdir}/_plot_{name}.{fmt}', **kwargs)

# type : list
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
default_linestyles = ['-','--','-.',':']
