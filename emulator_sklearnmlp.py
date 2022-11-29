# Command line arguments:
#   [1] file containing the training data
#   [2] optional -- fraction of data to use,
#                   can be used to check convergence

from sys import argv
import pickle

import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

VALIDATION_FRAC = 0.2
NUM_ROUND = 100

data_file = argv[1]
try :
    use_frac = float(argv[2])
except IndexError :
    use_frac = None

rng = np.random.default_rng(42)

with np.load(data_file) as f :
    param_names = list(f['param_names'])
    params = f['params']
    values = f['values'] # the log-likelihood

if use_frac is not None :
    select = rng.choice([True, False], size=len(values), p=[use_frac, 1.0-use_frac])
    params = params[select]
    values = values[select]

if False :
    select = values > -200
    params = params[select]
    values = values[select]

if False : # this doesn't really work
    values -= np.max(values)
    values = np.exp(values)

N = len(values)
print(f'{N} samples available')
assert N == params.shape[0]
assert len(param_names) == params.shape[1]

validation_select = rng.choice([True, False], size=N, p=[VALIDATION_FRAC, 1.0-VALIDATION_FRAC])

if False :
    min_clip = np.max(values) - 100
    values[values < min_clip] = min_clip

train_params = params[~validation_select]
train_values = values[~validation_select]
validation_params = params[validation_select]
validation_values = values[validation_select]

scaler = StandardScaler()
scaler.fit(train_params)
train_params = scaler.transform(train_params)
validation_params = scaler.transform(validation_params)

EPOCHS = 50
regr = MLPRegressor(hidden_layer_sizes=[256,]*8, activation='relu',
                    solver='adam', alpha=1e-3, batch_size=256,
                    learning_rate='invscaling', learning_rate_init=1e-2,
                    power_t=0.5,
                    max_iter=EPOCHS, random_state=42,
                    verbose=True)

stdev = np.std(train_params, axis=0)

for ii in range(EPOCHS) :

    if True :
        train_params_ = train_params + rng.normal(0.0, 0.1*stdev, train_params.shape)
    else :
        train_params_ = train_params
    
    regr = regr.partial_fit(train_params_, train_values)
    print(f'{regr.score(train_params, train_values)} \t {regr.score(validation_params, validation_values)}')


ypred = regr.predict(validation_params)

np.savez('test_sklearnmlp.npz',
         truth=validation_values,
         predictions=ypred)

with open('model_sklearnmlp.pkl', 'wb') as f :
    pickle.dump(regr, f)
