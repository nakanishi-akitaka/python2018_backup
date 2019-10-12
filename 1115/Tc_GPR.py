# -*- coding: utf-8 -*-
"""
Hydride Tc Regression

1. Hydride Tc Regression (GPR)
2. Applicability Domain (k-NN)
3. Double Cross Validation
4. y-randamization

Parameters
----------
Nothing

Returns
-------
Nothing

Input file
----------
tc_train.csv:
    Tc, atomic number 1&2, the number of atoms 1&2, pressure 
    of already calculated materials

tc_test.csv:
    Tc, atomic number 1&2, the number of atoms 1&2, pressure 
    of XnHm (n,m=1,...,10): X=He~At (without rare gas)

Outnput file
------------
Tc_EN_AD_DCV.csv:
    chemical formula, P, Tc, AD

-----------------------------------
Created on Wed Nov  7 13:53:42 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from my_library              import ad_knn
from sklearn.gaussian_process         import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def read_fxy_csv(name): 
    data = np.array(pd.read_csv(filepath_or_buffer=name, index_col=0,
                                header=0, sep=','))[:,:]
    f = np.array(data[:,0],dtype=np.unicode)
    y = np.array(data[:,1],dtype=np.float)
    X = np.array(data[:,2:],dtype=np.float)
    return f, X, y

start = time()

name = 'GPR'
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
output = 'Tc_' + name + '.csv'
print(name)
print(model)
print(output)

print()
print('read train & test data from csv file')
print()
train_file = 'tc_train.csv'
f_train, X_train, y_train = read_fxy_csv(train_file)
test_file = 'tc_test.csv'
f_test, X_test, y_test = read_fxy_csv(test_file)


iscaler=1
if(iscaler==1):
    scaler = StandardScaler()
else:
    scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
P_test = X_test[:,0]
X_test = scaler.transform(X_test)

model.fit(X_train, y_train)

from sklearn.metrics         import mean_absolute_error
from sklearn.metrics         import mean_squared_error
from sklearn.metrics         import r2_score
y_calc, _ = model.predict(X_train, return_std=True)
rmse  = np.sqrt(mean_squared_error (y_train, y_calc))
mae   =         mean_absolute_error(y_train, y_calc)
r2    =         r2_score           (y_train, y_calc)
print('C:  RMSE, MAE, R^2 = {:.3f}, {:.3f}, {:.3f}'\
.format(rmse, mae, r2))


import matplotlib.pyplot as plt
def yyplot(y_obs, y_pred):
    fig = plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.title("yy-plot")
    plt.scatter(y_obs, y_pred)
    y_all = np.concatenate([y_obs, y_pred])
    ylowlim = np.amin(y_all) - 0.05 * np.ptp(y_all)
    yupplim = np.amax(y_all) + 0.05 * np.ptp(y_all)
    plt.plot([ylowlim, yupplim],
             [ylowlim, yupplim],'k-')
    plt.ylim( ylowlim, yupplim)
    plt.xlim( ylowlim, yupplim)
    plt.xlabel("y_observed")
    plt.ylabel("y_predicted")
    
    plt.subplot(1,2,2)
    error = np.array(y_pred-y_obs)
    plt.hist(error)
    plt.title("Error histogram")
    plt.xlabel('prediction error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    return fig

yyplot(y_train, y_calc)

# Prediction,
y_pred, y_std = model.predict(X_test, return_std=True)

# Applicability Domain (inside: +1, outside: -1)
y_appd = ad_knn(X_train, X_test)

data = []
for i in range(len(X_test)):
    temp = (f_test[i], int(P_test[i]), int(y_pred[i]), y_std[i], y_appd[i])
    data.append(temp)

properties=['formula','P', 'Tc', 'std', 'AD']
df = pd.DataFrame(data, columns=properties)
df.sort_values('Tc', ascending=False, inplace=True)

# df.to_csv(output, index=False)
df_in_ = df[df.AD ==  1]
df_in_.to_csv(output, index=False)
print('Predicted Tc is written in file {}'.format(output))

niter=10
scores = np.zeros((niter,3))
for iiter in range(niter):
    y_rand = np.random.permutation(y_train)
    model.fit(X_train, y_rand)
    y_pred = model.predict(X_train)
    rmse  = np.sqrt(mean_squared_error (y_rand, y_pred))
    mae   =         mean_absolute_error(y_rand, y_pred)
    r2    =         r2_score           (y_rand, y_pred)
    scores[iiter,:] = np.array([rmse,mae,r2])
means, stds = np.mean(scores, axis=0),np.std(scores, axis=0)
print()
print("y-randomization")
print('In {:} iterations, average +/- standard deviation'.format(niter))
print('RMSE: {:.3f} (+/-{:.3f})'.format(means[0], stds[0]))
print('MAE: {:.3f} (+/-{:.3f})'.format(means[1], stds[1]))
print('R^2: {:.3f} (+/-{:.3f})'.format(means[2], stds[2]))

print('rnd:RMSE, MAE, R^2 = {:.3f}, {:.3f}, {:.3f} (ave)'\
      .format(means[0], means[1], means[2]))
print('rnd:RMSE, MAE, R^2 = {:.3f}, {:.3f}, {:.3f} (std)'\
      .format(stds[0], stds[1], stds[2]))

# print(X_train[:10])
print('{:.2f} seconds '.format(time() - start))
