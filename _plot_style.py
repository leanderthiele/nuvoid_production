# Import this to get a uniform style

from matplotlib import pyplot as plt

plt.style.use('dark_background')

def savefig (fig, name) :
    fmt = 'png'
    kwargs = dict(bbox_inches='tight', transparent=False)
    outdir = '.'

    fig.savefig(f'{outdir}/_plot_{name}.{fmt}', **kwargs)
