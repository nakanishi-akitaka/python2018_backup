# -*- coding: utf-8 -*-
"""
Calculate parameters from physical properties

Created on Tue Jul 10 15:51:58 2018

@author: Akitaka

Ref:
"A Data-Driven Statistical Model for Predicting the Critical Temperature
 of a Superconductor"
https://arxiv.org/abs/1803.10260
1.Atomic Mass atomic, mass units (AMU)
2.First Ionization Energy, kilo-Joules per mole (kJ/mol)
3.Atomic Radius, picometer (pm)
4.Density, kilograms per meters cubed (kg/m3)
5.Electron Affinity, kilo-Joules per mole (kJ/mol)
6.Fusion Heat, kilo-Joules per mole (kJ/mol)
7.Thermal Conductivity, watts per meter-Kelvin (W/(m ﾃ・K))
8.Valence no units typical number of chemical bonds formed by the element
Table 1: This table shows the properties of an element which are used for creating features to
predict Tc.

Mean = ﾂｵ = (t1 + t2)/2 
Weighted mean = ﾎｽ = (p1t1) + (p2t2) 
Geometric mean = (t1t2)**1/2
Weighted geometric mean = (t1)**p1*(t2)**p2
Entropy = 竏蜘1 ln(w1) 竏・w2 ln(w2)
Weighted entropy = 竏但 ln(A) 竏・B ln(B) 
Range = t1 竏・t2 (t1 > t2)
Weighted range = p1 t1 竏・p2 t2
Standard deviation = [(1/2)((t1 竏・ﾂｵ)**2 + (t2 竏・ﾂｵ)**2)]**1/2
Weighted standard deviation = [p1(t1 竏・ﾎｽ)**2 + p2(t2 竏・ﾎｽ)**2)]**1/2
Table 2:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from time import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# functions printing score
def print_score(y_test,y_pred):
    rmse  = np.sqrt(mean_squared_error (y_test,y_pred))
    mae   =         mean_absolute_error(y_test,y_pred)
    rmae  = np.sqrt(mean_squared_error (y_test,y_pred))/mae
    r2    =         r2_score           (y_test,y_pred)
    print('RMSE, MAE, RMSE/MAE, R^2 = {:.3f}, {:.3f}, {:.3f}, {:.3f}'\
    .format(rmse, mae, rmae, r2))

def print_gscv_score(gscv):
    print("Best parameters set found on development set:")
    print()
    print(gscv.best_params_)
    print()
#    print("Grid scores on development set:")
#    print()
#    means = gscv.cv_results_['mean_test_score']
#    stds = gscv.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
#        print("{:.3f} (+/-{:.03f}) for {:}".format(mean, std * 2, params))


print()
print('0, read dataset from csv file')
print()

file = "../0705/bandgapDFT_conv.csv"
data = np.array(pd.read_csv(file))[:,:]

X = []
y = []
for i in range(len(data)):
    if(data[i,81]>0):
        X.append(data[i,1:81])
        y.append(data[i,81])

X = np.array(X)
y = np.array(y)

# training data: y = sin(x) + noise
#
X_train = np.array(X)
y_train = np.array(y)
X_test = np.array(X)
y_test = np.array(y)

#%%     
start = time()
print('')
print('')
print('# 1. SVR with default hyper parameters')
# step 1. model
mod = SVR() 

# step 2. learning
mod.fit(X_train, y_train)
y_pred = mod.predict(X_train)
print('train data: ',end="")
print_score(y_train, y_pred)

# step 3. test
y_pred = mod.predict(X_test)
print('test  data: ',end="")
print_score(y_test,  y_pred)
print('{:.2f} seconds '.format(time() - start))

# step 4. visualize outputs
#plt.scatter(i, y_test,  color='black', label='test data')
#plt.plot(i, y_pred, color='blue', linewidth=3, label='model')
#plt.xlabel('data')
#plt.ylabel('target')
#plt.legend()
#plt.show()
# yy-plot
plt.rcParams["font.size"] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_test, y_pred)
max_y = np.max(np.array([y_test, y_pred]))
min_y = np.min(np.array([y_test, y_pred]))
plt.plot([min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y)],
         [min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y)],'k-')
plt.ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
plt.xlim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
plt.xlabel("y_observed")
plt.ylabel("y_predicted")
plt.show()

#%%
start = time()
print('')
print('')
print('# 2. parameter optimization (Grid Search)')

# step 1. model
mod = SVR() 

# step 2. learning with optimized parameters
# search range
# https://datachemeng.com/fastoptsvrhyperparams/
# range_c = 2**np.arange(  -5, 11, dtype=float)
# range_e = 2**np.arange( -10,  1, dtype=float)
# range_g = 2**np.arange( -20, 11, dtype=float)
# 1.gamma = miximize gram matrix
# 2.optimize only epsilon with C = 3 (when X is autoscaled) ang opted gamma
# 3.optimize only C with opted epsilon ang opted gamma
# 4.optimize only gamma with opted C and epsilon
range_c = 2**np.arange(  -5, 11, dtype=float)
range_e = 2**np.arange( -10,  1, dtype=float)
range_g = 2**np.arange( -20, 11, dtype=float)
print()
print('Search range')
print('c = ', range_c)
print('g = ', range_g)
print('e = ', range_e)
print()

# Set the parameters by cross-validation
param_grid = [{'kernel': ['rbf'],
               'gamma': range_g,
               'C': range_c,
               'epsilon': range_e}]

from sklearn.model_selection import KFold
cv_kf = KFold(n_splits=2, shuffle=True)
# gscv = GridSearchCV(mod, param_grid, cv=cv_kf, scoring='neg_mean_absolute_error')
# gscv = GridSearchCV(mod, param_grid, cv=cv_kf, scoring='neg_mean_squared_error')
gscv = GridSearchCV(mod, param_grid, cv=cv_kf, scoring='r2')
gscv.fit(X_train, y_train)

y_pred = gscv.predict(X_train)
print_gscv_score(gscv)
print('train data: ',end="")
print_score(y_train, y_pred)

# step 3. test
y_pred = gscv.predict(X_test)
print('test  data: ',end="")
print_score(y_test,  y_pred)
print('{:.2f} seconds '.format(time() - start))

# step 4. visualize outputs
#plt.scatter(i, y_test,  color='black', label='test data')
#plt.plot(i, y_pred, color='blue', linewidth=3, label='model')
#plt.xlabel('data')
#plt.ylabel('target')
#plt.legend()
#plt.show()

# yy-plot
plt.rcParams["font.size"] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_test, y_pred)
max_y = np.max(np.array([y_test, y_pred]))
min_y = np.min(np.array([y_test, y_pred]))
plt.plot([min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y)],
         [min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y)],'k-')
plt.ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
plt.xlim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
plt.xlabel("y_observed")
plt.ylabel("y_predicted")
plt.show()

#}}}

