# -*- coding: utf-8 -*-
"""
  Hydride Tc
+ Support Vector Machine (Regression)
+ One-Class Support Vector Machine (Applicability Domain)

Created on Thu Jul 26 09:19:06 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
# from matplotlib              import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.svm             import SVR
from sklearn.svm             import OneClassSVM
from pymatgen                import periodic_table, Composition
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from my_library              import print_gscv_score
from my_library              import print_score
from my_library              import yyplot
from my_library              import dcv

def read_xy_csv(name): 
    data = np.array(pd.read_csv(filepath_or_buffer=name,
                                index_col=0, header=0, sep=','))[:,:]
    y = data[:,0]
    X = data[:,1:]
    return X, y

print(__doc__)

start = time()

print('')
print('read train & test data from csv file')
print('')
train_file = 'tc_train.csv'
X_train, y_train = read_xy_csv(train_file)
test_file = 'tc_test.csv'
X_test, y_test = read_xy_csv(test_file)

# print statistics of database
ltest=False
if(ltest):
    data = pd.read_csv(filepath_or_buffer='tc_train.csv',
                       index_col=0, header=0, sep=',')
    data.drop('Z2', axis=1, inplace=True)
    print(data.describe())

#%%

# range_c = 2**np.arange( -5,  11, dtype=float)
# range_e = 2**np.arange( -10,  1, dtype=float)
# range_g = 2**np.arange( -20, 11, dtype=float)
range_c = 2**np.arange(   7, 10, dtype=float)
range_e = 2**np.arange(  -3,  1, dtype=float)
range_g = 2**np.arange(   7, 10, dtype=float)

print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('e = ', range_e[0], ' ... ',range_e[len(range_e)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

# Set the parameters by cross-validation
scaler = MinMaxScaler()
scaler = StandardScaler()
mod = SVR()
pipe = Pipeline([
('scaler', scaler),
('svr', mod)
])

param_grid = [
{'svr__kernel': ['rbf'], 'svr__gamma': range_g,
 'svr__C': range_c,'svr__epsilon': range_e},
]
n_splits = 5 
cv = ShuffleSplit(n_splits=n_splits, test_size=0.2)
cv = KFold(n_splits=n_splits, shuffle=True)
score='neg_mean_absolute_error'

gscv = GridSearchCV(pipe, param_grid, cv=cv, scoring=score)
gscv.fit(X_train, y_train)
print_gscv_score(gscv)

y_pred = gscv.predict(X_train)
print('train data: ',end="")
print_score(y_train, y_pred)
# visualize
fig = yyplot(y_train, y_pred)

# Novelty detection by One Class SVM with optimized hyperparameter
clf = OneClassSVM(nu=0.10, kernel=gscv.best_params_['svr__kernel'],
  gamma=gscv.best_params_['svr__gamma'])
clf.fit(X_train)

y_pred = gscv.predict(X_test)    # prediction
reliability = clf.predict(X_test) # outliers = -1

data = []
output = 'test2.csv'
for i in range(len(X_test)):
    satom1 = periodic_table.get_el_sp(int(X_test[i][0])) 
    satom2 = periodic_table.get_el_sp(int(X_test[i][1])) 
    natom1 = int(X_test[i][2])
    natom2 = int(X_test[i][3])
    str_mat = str(satom1) + str(natom1) + str(satom2) + str(natom2)
    formula = Composition(str_mat).reduced_formula
    temp = (formula, int(X_test[i][4]), int(y_pred[i]), reliability[i])
    data.append(temp)

properties=['formula','P', 'Tc', 'AD']
df = pd.DataFrame(data, columns=properties)
df.sort_values('Tc', ascending=False, inplace=True)
df_in_ = df[df.AD ==  1]
df_in_.to_csv(output, index=False)
print('Predicted Tc is written in file {}'.format(output))

#%%
param_grid = [
{'kernel': ['rbf'], 'gamma': range_g,
 'C': range_c,'epsilon': range_e},
]
for i in range(10):
    dcv(X_train, y_train, mod, param_grid)

print('{:.2f} seconds '.format(time() - start))
