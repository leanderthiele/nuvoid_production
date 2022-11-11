""" Small utility to inspect an optuna run
Command line arguments:
    [1] sim version
    [2] sim index
    [3] (optional) additional version string
"""

from sys import argv
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
import optuna

sim_version = argv[1]
sim_index = int(argv[2])
try :
    vstr = '_' + argv[3]
except IndexError :
    vstr = ''

diagnostics = ['slice', 'edf', 'optimization_history',
               'parallel_coordinate', 'param_importances', ]
for d in diagnostics :
    exec(f'from optuna.visualization import plot_{d}')

study = optuna.load_study(study_name=f'hod_fit{vstr}_{sim_version}_{sim_index}',
                          storage='mysql://optunausr:pwd@tigercpu:3310/optunadb'\
                                  '?unix_socket=/home/lthiele/mysql/mysql.sock')

CUTOFF = 20

for state in optuna.trial.TrialState :
    print(f'{len(study.get_trials(states=(state,), deepcopy=False))} {str(state).split(".")[-1]} trials')
best_trial = study.best_trial
print(f'best trial:\n'\
      f'\tobjective = {best_trial.values[0]},\n'\
      f'\thash = {best_trial.user_attrs["hod_hash"]},\n'
      f'\tparams = {best_trial.params}')

VIDE_OUT = 'untrimmed_dencut'
RMIN = 30
RMAX = 80
NBINS = 32
def get_hist(path) :
    datafiles = glob(f'{path}/{VIDE_OUT}_centers_central_*.out')
    assert len(datafiles) == 1
    r = np.loadtxt(datafiles[0], usecols=4)
    h, e = np.histogram(r, bins=NBINS, range=(RMIN, RMAX))
    return h, 0.5*(e[1:]+e[:-1])
h_dat, c_dat = get_hist('/tigress/lthiele/boss_dr12/voids/sample_test')
sim_samples = glob(f'/scratch/gpfs/lthiele/nuvoid_production/{sim_version}_{sim_index}/hod/{best_trial.user_attrs["hod_hash"]}/sample_*')
h_sim = []
for sample in sim_samples :
    h_, c_ = get_hist(sample)
    assert np.allclose(c_dat, c_)
    h_sim.append(h_)
h_sim = np.mean(np.array(h_sim), axis=0)
fig, ax = plt.subplots(nrows=2)
ax_vsf = ax[0]
ax_diff = ax[1]
ax_vsf.scatter(c_dat, h_dat, label='CMASS')
ax_vsf.scatter(c_dat, h_sim, label=f'simulation (mean {len(sim_samples)} augments)')
ax_vsf.set_yscale('log')
ax_vsf.set_ylabel('counts')
ax_vsf.set_title('VSF for best-fit HOD')
ax_vsf.legend(loc='lower left')
ax_vsf.text(0.95, 0.95, f'$-\log\mathcal{{L}}$={best_trial.values[0]:.2f}',
            transform=ax_vsf.transAxes, va='top', ha='right')
ax_diff.scatter(c_dat, (h_sim-h_dat)/np.sqrt(h_sim))
ax_diff.set_ylabel('$\Delta/\sqrt{{\sf sim}}$')
ax_diff.set_xlabel('R [Mpc/h]')
ax_diff.axhline(0, color='black', linestyle='dashed')
fig.savefig(f'hod_fit{vstr}_{sim_version}_{sim_index}_bestfitvsf.pdf', bbox_inches='tight')


for d in diagnostics :
    fname = f'hod_fit{vstr}_{sim_version}_{sim_index}_{d}.pdf'
    print(fname)
    if d in ['optimization_history', 'slice', ] :
        obj = lambda t: t.values[0] if t.values[0]<CUTOFF else CUTOFF
    else :
        obj = None
    exec(f'fig = plot_{d}(study, target=obj)')
    fig.write_image(fname)
