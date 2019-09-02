# -*- coding: utf-8 -*-
"""
  Hydride Tc
+ Support Vector Machine (Regression)
+ One-Class Support Vector Machine (Applicability Domain)

Created on Thu Aug  2 15:41:37 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
# from matplotlib              import pyplot as plt
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.svm             import SVR, OneClassSVM
from pymatgen                import periodic_table, Composition
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from my_library              import print_gscv_score, print_score_rgr
from my_library              import yyplot, dcv, optimize_gamma

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
range_c = 2**np.arange(  -5+10, 11, dtype=float)
range_e = 2**np.arange( -10+5,  1, dtype=float)
range_g = 2**np.arange( -20+25, 11, dtype=float)

print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('e = ', range_e[0], ' ... ',range_e[len(range_e)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

# Set the parameters by cross-validation
scaler = MinMaxScaler()
scaler = StandardScaler()
model = SVR()
pipe = Pipeline([
('scaler', scaler),
('model', model)
])

param_grid = [
{'model__kernel': ['rbf'], 'model__gamma': range_g,
 'model__C': range_c,'model__epsilon': range_e},
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
print_score_rgr(y_train, y_pred)
# visualize
fig = yyplot(y_train, y_pred)

#%%
# Novelty detection by One Class SVM with optimized hyperparameter
clf = OneClassSVM(nu=0.003, kernel=gscv.best_params_['model__kernel'],
  gamma=gscv.best_params_['model__gamma'])
clf.fit(X_train)
reliability1 = clf.predict(X_test) # outliers = -1

# Novelty detection by One Class SVM with optimized hyperparameter
optgamma = optimize_gamma(X_train, range_g) 
clf = OneClassSVM(nu=0.003, kernel=gscv.best_params_['model__kernel'],
  gamma=optgamma)
clf.fit(X_train)
reliability2 = clf.predict(X_test) # outliers = -1

print("gamma1, 2 = ", gscv.best_params_['model__gamma'], optgamma)

y_pred = gscv.predict(X_test)    # predicted y

data = []
for i in range(len(X_test)):
    satom1 = periodic_table.get_el_sp(int(X_test[i][0])) 
    satom2 = periodic_table.get_el_sp(int(X_test[i][1])) 
    natom1 = int(X_test[i][2])
    natom2 = int(X_test[i][3])
    str_mat = str(satom1) + str(natom1) + str(satom2) + str(natom2)
    formula = Composition(str_mat).reduced_formula
    temp = (formula, int(X_test[i][4]), int(y_pred[i]), reliability1[i],
            reliability2[i])
    data.append(temp)

properties=['formula','P', 'Tc', 'AD1', 'AD2']
df = pd.DataFrame(data, columns=properties)
df.sort_values('Tc', ascending=False, inplace=True)

output = 'test2_Tc_SVM_OCSVM_DCV.csv'
# df.to_csv(output, index=False)
df_in_ = df[(df.AD1 ==  1) | (df.AD2 ==  1)]
df_in_.to_csv(output, index=False)

print('Predicted Tc is written in file {}'.format(output))

#%%
param_grid = [
{'kernel': ['rbf'], 'gamma': range_g, 'C': range_c,'epsilon': range_e},
]
for i in range(1):
    dcv(X_train, y_train, model, param_grid)

print('{:.2f} seconds '.format(time() - start))
