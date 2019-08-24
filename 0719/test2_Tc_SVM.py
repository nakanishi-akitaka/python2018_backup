# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:34:01 2018

@author: Akitaka
"""

# -*- coding: utf-8 -*-
"""
ML of Tc by SVM (0426)

Created on Thu Jul 19 12:54:41 2018

@author: Akitaka
"""
import pandas as pd
import numpy as np
from time import time
from matplotlib import pyplot as plt
from pymatgen import Element
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#
# function print score of learning and prediction 
#
def print_score(y_test,y_pred):
    rmse  = np.sqrt(mean_squared_error (y_test,y_pred))
    mae   =         mean_absolute_error(y_test,y_pred)
    rmae  = np.sqrt(mean_squared_error (y_test,y_pred))/mae
    r2    =         r2_score           (y_test,y_pred)
    print('RMSE, MAE, RMSE/MAE, R^2 = {:.3f}, {:.3f}, {:.3f}, {:.3f}'\
    .format(rmse, mae, rmae, r2))
# ref: RMSE/MAE
# https://funatsu-lab.github.io/open-course-ware/basic-theory/accuracy-index/

#
# function print score of grid search
#
def print_gscv_score(gscv): #{{{
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

#
# function print high Tc 
# {{{
def print_high_Tc(X_test,y_pred):
    path_w = 'test1_tc.txt'
    with open(path_w, mode='w') as f:
        for i in range(len(X_test)):
            if(y_pred[i]> 100):
                satom0 = Element.from_Z(int(X_test[i][0])).symbol.lstrip() 
                satom1 = Element.from_Z(int(X_test[i][1])).symbol.lstrip() 
                natom0 = int(X_test[i][2])
                natom1 = int(X_test[i][3])
                p  = int(X_test[i][4])
                tc = int(y_pred[i])
                f.write('{:>2}{}{}{} P = {:>3} GPa Tc = {} K \n'
                .format(satom0,natom0,satom1,natom1,p,tc))  
    print('Predicted Tc is written in file {}'.format(path_w))
#}}}
#
# function Observed-Predicted Plot (yyplot) 
#
def yyplot(y_obs,y_pred):
    yvalues = np.concatenate([y_obs, y_pred])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    fig = plt.figure(figsize=(8,8))
    plt.scatter(y_obs, y_pred)
    plt.plot([ymin-yrange*0.01, ymax+yrange*0.01],[ymin-yrange*0.01, ymax+yrange*0.01])
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('y_observed', fontsize=24)
    plt.ylabel('y_predicted', fontsize=24)
    plt.title('Observed-Predicted Plot', fontsize=24)
    plt.tick_params(labelsize=16)
    plt.show()
    return fig

def read_file(name): 
    data = np.array(pd.read_csv(filepath_or_buffer=name,
                               header=None,sep=','))[:,:]
    y = data[:,0]
    X = data[:,1:]
    return X, y

print(__doc__)
start = time()

train_file = 'tc_train.csv'
X, y = read_file(train_file)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



print('')
print('# SVR with GridSearched hyper parameters after MinMaxScaler')
print('')
# step 1. model using pipeline
pipe = Pipeline([
('scaler', MinMaxScaler()),
('svr', SVR())
])

# estimation for different datasets = OK: MAE, NG: R^2
# http://univprof.com/archives/16-07-04-4453136.html
score='neg_mean_absolute_error'

# search range
# https://datachemeng.com/supportvectorregression/
# https://datachemeng.com/fastoptsvrhyperparams/
# range_c = 2**np.arange( -5,  11, dtype=float)
# range_e = 2**np.arange( -10,  1, dtype=float)
# range_g = 2**np.arange( -20, 11, dtype=float)
range_c = 2**np.arange(   9, 11, dtype=float)
range_e = 2**np.arange( -10,  1, dtype=float)
range_g = 2**np.arange(   9, 11, dtype=float)
param_grid = [
    {'svr__kernel': ['rbf'],'svr__C': range_c,
     'svr__epsilon': range_e, 'svr__gamma': range_g}]
print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('e = ', range_e[0], ' ... ',range_e[len(range_e)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
gscv = GridSearchCV(pipe, param_grid, scoring=score,cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score(gscv)

y_pred = gscv.predict(X_train)
print('train data: ',end="")
print_score(y_train, y_pred)

# step 4. visualize
# yy-plot (train)
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.title("yy-plot (train)")
plt.scatter(y_train, y_pred)
max_y = np.max(np.array([y_train, y_pred]))
min_y = np.min(np.array([y_train, y_pred]))
ylowlim = min_y - 0.05 * (max_y - min_y)
yupplim = max_y + 0.05 * (max_y - min_y)
plt.plot([ylowlim, yupplim],
         [ylowlim, yupplim],'k-')
plt.ylim( ylowlim, yupplim)
plt.xlim( ylowlim, yupplim)
plt.xlabel("y_observed")
plt.ylabel("y_predicted")

# Check: error follows a normal distribution?
plt.subplot(1,2,2)
error = np.array(y_pred-y_train)
plt.hist(error)
plt.title("Gaussian? (train)")
plt.xlabel('prediction error')
plt.ylabel('Frequency')
plt.show()

# step 3. predict
y_pred = gscv.predict(X_test)
print_high_Tc(X_test,y_pred)
print('{:.2f} seconds '.format(time() - start))
# }}}