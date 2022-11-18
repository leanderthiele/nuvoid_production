# Command line arguments:
#   [1] file containing the training data

from sys import argv

import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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
if False :
    min_clip = np.max(values) - 100
    values[values < min_clip] = min_clip

train_params = params[~validation_select]
train_values = values[~validation_select]
validation_params = params[validation_select]
validation_values = values[validation_select]

if False :
    train_params += rng.normal(0.0, 0.1*np.std(train_params, axis=0), train_params.shape)

scaler = StandardScaler()
scaler.fit(train_params)
train_params = scaler.transform(train_params)
validation_params = scaler.transform(validation_params)

regr = MLPRegressor(hidden_layer_sizes=[64,]*4, activation='relu',
                    solver='adam', alpha=1e-4, batch_size=64,
                    learning_rate='constant', learning_rate_init=1e-3,
                    max_iter=10,
                    verbose=True).fit(train_params, train_values)

ypred = regr.predict(validation_params)

np.savez('test_sklearnmlp.npz',
         truth=validation_values,
         predictions=ypred)
