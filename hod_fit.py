""" Use optuna to fit the HOD by maximizing the VSF likelihood
Command line arguments:
    [1] simulation version ('cosmo_varied' or 'fiducial')
    [2] simulation index
"""

import sys
from sys import argv
import subprocess
import logging
import hashlib
import random
from time import sleep

import optuna
from optuna.samplers import TPESampler

codebase = '/home/lthiele/nuvoid_production'

# small custom exception for VIDE failures
class VIDEFailure(RuntimeError) :
    pass

# custom exception for SIGSEGV
class SegfaultFailure(RuntimeError) :
    pass

# custom exception for OOM
class OOMFailure(RuntimeError) :
    pass

# invoked when there are repeated failures
# (indicating that something more fundamental is going wrong)
class RepeatedFailure(RuntimeError) :
    pass

class Objective :

    def __init__(self, sim_version, sim_index) :
        """ Constructor
        sim_version ... 'cosmo_varied' or 'fiducial'
        sim_index ... index of the simulation, integer
        """
        self.wrk_dir = f'/scratch/gpfs/lthiele/nuvoid_production/{sim_version}_{sim_index}'
        self.failures = []

    def _draw_hod(self, trial) :
        args = ''

        # clear evidence that Rockstar works better
        args += ' cat=rockstar'

        # can't hurt to include assembly bias, seems like T/U works better
        args += ' secondary=kinpot'
        transfP1 = trial.suggest_float('transfP1', -3, 3)
        args += f' hod_transfP1={transfP1}'
        abias = trial.suggest_float('abias', -1, 1)
        args += f' hod_abias={abias}'

        # some of these I have adjusted to narrower ranges after first trial runs
        log_Mmin = trial.suggest_float('log_Mmin', 12.6, 13.2)
        args += f' hod_log_Mmin={log_Mmin}'
        sigma_logM = trial.suggest_float('sigma_logM', 0.15, 0.5)
        args += f' hod_sigma_logM={sigma_logM}'
        log_M0 = trial.suggest_float('log_M0', 12.5, 15.0)
        args += f' hod_log_M0={log_M0}'
        log_M1 = trial.suggest_float('log_M1', 12.5, 15.0)
        args += f' hod_log_M1={log_M1}'
        alpha = trial.suggest_float('alpha', 0.3, 0.8)
        args += f' hod_alpha={alpha}'

        # can't hurt to include velocity bias
        args += ' have_vbias=True'
        transf_eta_cen = trial.suggest_float('transf_eta_cen', 0.0, 10.0)
        args += f' hod_transf_eta_cen={transf_eta_cen}'
        transf_eta_sat = trial.suggest_float('transf_eta_sat', -1, 1)
        args += f' hod_transf_eta_sat={transf_eta_sat}'

        # redshift dependence
        # Range in scale factor around pivot is around 0.05,
        # mu times this number should give a reasonable variation
        args += ' have_zdep=True'
        mu_Mmin = trial.suggest_float('mu_Mmin', -10.0, 10.0)
        args += f' hod_mu_Mmin={mu_Mmin}'
        mu_M1 = trial.suggest_float('mu_M1', -40.0, 40.0)
        args += f' hod_mu_M1={mu_M1}'

        return args

    def __call__(self, trial) :
        hod_args = self._draw_hod(trial)
        hod_hash = hashlib.md5(hod_args.encode('utf-8')).hexdigest()
        trial.set_user_attr('hod_hash', hod_hash) # useful to have this back-reference
        result = subprocess.run(f'bash {codebase}/hod_like.sh {self.wrk_dir} {hod_hash} {hod_args}',
                                shell=True, check=False)

        if result.returncode == 0 :
            # terminated successfully
            with open(f'{self.wrk_dir}/hod/{hod_hash}/loglike.info', 'r') as f :
                line = f.readline().strip()
                line = line.split('=')
                assert line[0] == 'loglike_tot'
                loglike = float(line[1])
            self.failures = [] # reset
            return -loglike

        # maybe useful to document
        trial.set_user_attr('returncode', result.returncode)

        self.failures.append((hod_hash, result.returncode))
        if len(self.failures) > 2 : # we can tolerate 2 random failures in a row, but not more
            raise RepeatedFailure(self.failures)

        if result.returncode == 42 :
            # this is our magic returncode for random VIDE failures, we understand that this happens
            # in some rare cases ...
            raise VIDEFailure

        if result.returncode == 137 :
            # out of memory, this should happen very rarely and is usually sign of bad input parameters
            raise OOMFailure

        if result.returncode == 139 :
            # segmentation fault, not sure why this sometimes happens
            raise SegfaultFailure

        result.check_returncode()

# driver
if __name__ == '__main__' :
    
    sim_version = argv[1]
    sim_index = int(argv[2])

    # avoid many simultaneous calls to database on startup
    sleep(random.uniform(0, 10))

    # set up optuna logging
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))

    # set up our study
    # options to TPESampler added over time from docs
    study = optuna.create_study(sampler=TPESampler(n_startup_trials=80,
                                                   constant_liar=True,
                                                   multivariate=True),
                                study_name=f'hod_fit_v2_{sim_version}_{sim_index}',
                                storage='mysql://optunausr:pwd@tigercpu:3310/optunadb'\
                                        '?unix_socket=/home/lthiele/mysql/mysql.sock',
                                directions=['minimize', ],
                                load_if_exists=True)

    # create our objective callable
    objective = Objective(sim_version, sim_index)

    # run optuna
    study.optimize(objective, catch=(VIDEFailure, OOMFailure, SegfaultFailure,))
