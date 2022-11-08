""" Small utility to inspect an optuna run
Command line arguments:
    [1] optuna study name
"""

from sys import argv

study_name = argv[1]

import optuna

diagnostics = ['contour', 'edf', 'intermediate_values', 'optimization_history',
               'parallel_coordinate', 'param_importances', 'slice', ]
for d in diagnostics :
    exec(f'from optuna.visualization.matplotlib import plot_{d}')

study = optuna.load_study(study_name=study_name,
                          storage='mysql://optunausr:pwd@tigercpu:3310/optunadb'\
                                  '?unix_socket=/home/lthiele/mysql/mysql.sock')

for d in diagnostics :
    exec(f'fig = plot_{d}(study)')
    fig.savefig(f'{study_name}_{d}.pdf', bbox_inches='tight')
