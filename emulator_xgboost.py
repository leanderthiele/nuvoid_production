# Command line arguments:
#   [1] file containing the training data

from sys import argv

import numpy as np

import xgboost as xgb

VALIDATION_FRAC = 0.2
NUM_ROUND = 100

data_file = argv[1]

with np.load(data_file) as f :
    param_names = list(f['param_names'])
    params = f['params']
    values = f['values'] # the log-likelihood

# FIXME
# values = np.exp(values-np.max(values))

N = len(values)
assert N == params.shape[0]
assert len(param_names) == params.shape[1]

rng = np.random.default_rng(42)
validation_select = rng.choice([True, False], size=N, p=[VALIDATION_FRAC, 1.0-VALIDATION_FRAC])

# TODO maybe this helps
min_clip = np.max(values) - 100
values[values < min_clip] = min_clip

train_params = params[~validation_select]
train_values = values[~validation_select]
validation_params = params[validation_select]
validation_values = values[validation_select]

# to increase size of training set and learn what we are actually interested in,
# we concentrate on the *difference* in log-likelihoods
train_param_diffs = (train_params[:, None, :] - train_params[None, :, :]).reshape(-1, train_params.shape[1])
train_values = (train_values[:, None] - train_values[None, :]).flatten()
validation_param_diffs = (train_params[:, None, :] - train_params[None, :, :]).reshape(-1, train_params.shape[1])
validation_values = (train_values[:, None] - train_values[None, :]).flatten()

train_params = np.concatenate((np.repeat(train_params, train_params.shape[0], axis=0),
                               train_param_diffs), axis=-1)
validation_params = np.concatenate((np.repeat(validation_params, validation_params.shape[0], axis=0),
                                   validation_param_diffs), axis=-1)

# TODO maybe this helps
# train_params += rng.normal(loc=0, scale=0.001*np.std(train_params, axis=0),
#                            size=train_params.shape)

dtrain = xgb.DMatrix(train_params, label=train_values)
dvalidation = xgb.DMatrix(validation_params, label=validation_values)

xgb.set_config(verbosity=2)

xgb_params = {
              'max_depth': 6,
              'eta': 0.3,
              'min_child_weight': 10,
              'gamma': 0.5,
              'subsample': 0.7,
              'alpha': 100,
              'lambda': 100,
              'objective': 'reg:squarederror',
#              'huber_slope': 1,
              'eval_metric': 'rmse',
              'nthread': 4,
             }

evallist = [(dtrain, 'train'), (dvalidation, 'eval')]

bst = xgb.train(xgb_params, dtrain, NUM_ROUND, evallist,
                verbose_eval=True)

bst.save_model('test.model')

validation_predict = bst.predict(dvalidation)
np.savez('test_valid.npz',
         param_names=param_names,
         params=validation_params,
         values=validation_predict,
         truths=validation_values)
