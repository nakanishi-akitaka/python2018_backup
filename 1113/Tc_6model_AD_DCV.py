# -*- coding: utf-8 -*-
"""
Hydride Tc Regression

1. Hydride Tc Regression (RR, LASSO, EN, k-NN, RF, SVM)
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
Created on Mon Nov  5 16:20:13 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.svm             import SVR
from sklearn.linear_model    import Ridge, Lasso, ElasticNet
from pymatgen                import periodic_table, Composition
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from my_library              import print_gscv_score_rgr, dcv_rgr
from my_library              import y_randamization_rgr, ad_knn, ad_ocsvm


def read_xy_csv(name): 
    data = np.array(pd.read_csv(filepath_or_buffer=name,
                                index_col=0, header=0, sep=','))[:,:]
    y = data[:,0]
    X = data[:,1:]
    return X, y

start = time()

range_f =  0.1*np.arange(  1, 11, dtype=int)
range_n = np.arange( 1,  10, dtype=int)
range_c = 2**np.arange(  -5+15, 11, dtype=float)
range_e = 2**np.arange( -10+5,  1, dtype=float)
range_g = 2**np.arange( -20+30, 11, dtype=float)


test = {
'RR'   :{'name':'Ridge Regression',
         'model':Ridge(),
         'param':{'model__alpha':range_f}},
'EN'   :{'name':'Elastic Net     ',
         'model':ElasticNet(),
         'param':{'model__alpha':range_f, 'model__l1_ratio':range_f}},
'LASSO':{'name':'LASSO           ',
         'model':Lasso(),
         'param':{'model__alpha':range_f}},
'kNN'  :{'name':'kNN             ',
         'model':KNeighborsRegressor(),
         'param':{'model__n_neighbors':range_n}},
'RF'   :{'name':'Random Forest',
         'model':RandomForestRegressor(),
         'param':{'model__max_features':range_f}},
'SVR'  :{'name':'SVR',
         'model':SVR(),
         'param':{'model__gamma': range_g, 
                  'model__C': range_c,
                  'model__epsilon': range_e}},
}


key = 'kNN' # 'RR' 'EN', 'LASSO', 'kNN', 'RF', 'SVR'
name = test[key]['name']
model = test[key]['model']
param_grid = test[key]['param']
output = 'Tc_' + key + '_AD_DCV.csv'
print(name)
print(model)
print(param_grid)
print(output)

print()
print('read train & test data from csv file')
print()
train_file = 'tc_train.csv'
X_train, y_train = read_xy_csv(train_file)
test_file = 'tc_test.csv'
X_test, y_test = read_xy_csv(test_file)
# print(X_train[:10])
# print statistics of database
if(False):
    data = pd.read_csv(filepath_or_buffer='tc_train.csv',
                       index_col=0, header=0, sep=',')
    data.drop('Z2', axis=1, inplace=True)
    print(data.describe())

# Set the parameters by cross-validation
scaler = MinMaxScaler()
scaler = StandardScaler()
pipe = Pipeline([('scaler', scaler),('model',  model)])

n_splits = 5 
cv = ShuffleSplit(n_splits=n_splits, test_size=0.2)
cv = KFold(n_splits=n_splits, shuffle=True)
gscv = GridSearchCV(pipe, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

# Prediction
y_pred = gscv.predict(X_test)

# Applicability Domain (inside: +1, outside: -1)
y_appd = ad_knn(X_train, X_test)
y_appd = ad_ocsvm(X_train, X_test)

data = []
for i in range(len(X_test)):
    satom1 = periodic_table.get_el_sp(int(X_test[i][0])) 
    satom2 = periodic_table.get_el_sp(int(X_test[i][1])) 
    natom1 = int(X_test[i][2])
    natom2 = int(X_test[i][3])
    str_mat = str(satom1) + str(natom1) + str(satom2) + str(natom2)
    formula = Composition(str_mat).reduced_formula
    temp = (formula, int(X_test[i][4]), int(y_pred[i]),y_appd[i])
    data.append(temp)

properties=['formula','P', 'Tc', 'AD']
df = pd.DataFrame(data, columns=properties)
df.sort_values('Tc', ascending=False, inplace=True)

# df.to_csv(output, index=False)
df_in_ = df[df.AD ==  1]
df_in_.to_csv(output, index=False)
print('Predicted Tc is written in file {}'.format(output))

niter=10
if(True):
    dcv_rgr(X_train, y_train, model, param_grid, niter)

if(True):
    y_randamization_rgr(X_train, y_train, model, scaler, param_grid, niter)

# print(X_train[:10])
print('{:.2f} seconds '.format(time() - start))
