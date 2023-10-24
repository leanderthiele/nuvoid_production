# Import this to get a uniform style

from matplotlib import pyplot as plt
from itertools import cycle

# plt.rcParams.update({'font.size': 20})
if False :
    plt.style.use('dark_background')
    black = 'white'
else :
    black = 'black'

def savefig (fig, name) :
    fmt = 'pdf'
#    kwargs = dict(bbox_inches='tight', transparent=True, dpi=200)
    kwargs = dict(bbox_inches='tight', transparent=False)
    outdir = '.' #'./desi_lunch_figures'

    fig.savefig(f'{outdir}/_plot_{name}.{fmt}', **kwargs)

# type : list
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
default_linestyles = ['-','--','-.',':']
