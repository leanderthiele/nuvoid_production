""" Script to add trials from a previous study to a new one,
used for different likelihood on the same void data.
Command line arguments:
    [1] old study name
    [3] directory where to find old studies
    [3] new study name
"""

from sys import argv
import subprocess

import optuna
from optuna.samplers import TPESampler

codebase = '/home/lthiele/nuvoid_production'

old_study_name = argv[1]
study_dir = argv[2]
new_study_name = argv[3]

old_study = optuna.load_study(study_name=old_study_name,
                              storage='mysql://optunausr:pwd@tigercpu:3310/optunadb'\
                                      '?unix_socket=/home/lthiele/mysql/mysql.sock')

new_study = optuna.create_study(sampler=TPESampler(n_startup_trials=0,
                                                   constant_liar=True,
                                                   multivariate=True),
                                study_name=new_study_name,
                                storage='mysql://optunausr:pwd@tigercpu:3310/optunadb'\
                                        '?unix_socket=/home/lthiele/mysql/mysql.sock',
                                directions=['minimize', ],
                                load_if_exists=True)

for old_trial in old_study.get_trials(states=(optuna.trial.TrialState.COMPLETE,), deepcopy=False) :
    hod_hash = old_trial.user_attrs['hod_hash']

    result = subprocess.run(f'bash {codebase}/hod_new_zsplit.sh {study_dir}/{hod_hash}',
                            shell=True, check=True)

    objective = -float(result.stdout.strip())

    new_trial = optuna.trial.create_trial(params=old_trial.params, distributions=old_trial.distributions,
                                          value=objective, user_attrs=old_trial.user_attrs)

    new_study.add_trial(new_trial)
