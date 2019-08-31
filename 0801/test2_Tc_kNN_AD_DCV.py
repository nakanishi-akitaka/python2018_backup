# -*- coding: utf-8 -*-
"""
1. Hydride Tc Regression (k-NN)
2. Reliability (k-NN)
3. Applicability Domain (k-NN)
4. Double Cross Validation
  Hydride Tc Regression

Created on Mon Jul 30 12:36:22 2018

@author: Akitaka
"""
import pandas as pd
import numpy as np
from time                    import time
# from matplotlib              import pyplot as plt
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.neighbors       import NearestNeighbors, KNeighborsRegressor
from pymatgen                import periodic_table, Composition
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from my_library              import print_gscv_score, print_score
from my_library              import yyplot, dcv

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

range_k = np.arange(  3, 11, dtype=int)

print()
print('Search range')
print('k = ', range_k[0], ' ... ',range_k[len(range_k)-1])
print()


# Set the parameters by cross-validation
scaler = MinMaxScaler()
scaler = StandardScaler()
model = KNeighborsRegressor()
pipe = Pipeline([
('scaler', scaler),
('model',  model)
])
param_grid = [{'model__n_neighbors':range_k}]

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

#%%
# Prediction
y_pred = gscv.predict(X_test)

# Applicability Domain
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_train)
dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
thr = dist.mean() - dist.std()
y_appd = 2 * (dist > thr) -1

# Standard Deviation (= Uncertainty <-> Reliability)
y_reli = np.std(y_train[neigh.kneighbors(X_test)[1]], axis=1)

data = []
output = 'test2.csv'
for i in range(len(X_test)):
    satom1 = periodic_table.get_el_sp(int(X_test[i][0])) 
    satom2 = periodic_table.get_el_sp(int(X_test[i][1])) 
    natom1 = int(X_test[i][2])
    natom2 = int(X_test[i][3])
    str_mat = str(satom1) + str(natom1) + str(satom2) + str(natom2)
    formula = Composition(str_mat).reduced_formula
    temp = (formula, int(X_test[i][4]), int(y_pred[i]), y_reli[i], y_appd[i])
    data.append(temp)

properties=['formula','P', 'Tc', 'Std', 'AD']
df = pd.DataFrame(data, columns=properties)
df.sort_values('Tc', ascending=False, inplace=True)
# df.to_csv(output, index=False)
df_in_ = df[df.AD ==  1]
df_in_.to_csv(output, index=False)
print('Predicted Tc is written in file {}'.format(output))

#%%
param_grid = [{'n_neighbors':range_k}]
for i in range(10):
    dcv(X_train, y_train, model, param_grid)

print('{:.2f} seconds '.format(time() - start))

