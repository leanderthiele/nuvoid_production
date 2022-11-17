# Command line arguments:
#   [1] file containing the training data

from sys import argv

import numpy as np

import xgboost as xgb

VALIDATION_FRAC = 0.2
NUM_ROUND = 10

data_file = argv[1]

with np.load(data_file) as f :
    param_names = list(f['param_names'])
    params = f['params']
    values = f['values'] # the log-likelihood

N = len(values)
assert N == params.shape[0]
assert len(param_names) == param_names.shape[1]

rng = np.random.default_rng()
validation_select = rng.choice([True, False], size=N, p=[VALIDATION_FRAC, 1.0-VALIDATION_FRAC])

dtrain = xgb.DMatrix(params[~validation_select], label=values[~validation_select])
dvalidation = xgb.DMatrix(params[validation_select], label=values[validation_select])

xgb_params = {
              'max_depth': 2,
              'eta': 1,
              'objective': 'reg:squarederror',
              'nthread': 4,
             }

evallist = [(dtrain, 'train'), (dtest, 'eval')]

bst = xgb.train(xgb_params, dtrain, NUM_ROUND, evallist,
                verbose_eval=True)
