# -*- coding: utf-8 -*-
"""
A.Make Tc csv files
1.convert tc.csv -> tc_data.csv
    chemical formula -> parameters(atomic No. & number of atoms)
    ex) S2H5  -> 16.0, 1.0, 2.0, 5.0
2.make tc_pred.csv
   all A_n H_m (n,m=1,...,10)

B.Hydride Tc Regression
1.Regression
2.Applicability Domain

Parameters
----------
Nothing

Returns
-------
Nothing

Input file
----------
tc.csv:

Temporary file
----------
tc_data.csv:
    Tc, atomic number 1&2, the number of atoms 1&2, pressure 
    of already calculated materials

tc_pred.csv:
    Tc, atomic number 1&2, the number of atoms 1&2, pressure 
    of XnHm (n,m=1,...,10): X=He~At (without rare gas)

Outnput file
------------
tc_'model_name'.csv:
    chemical formula, P, Tc, AD

-----------------------------------
Created on Tue Nov 13 13:18:00 2018

@author: Akitaka
"""

import numpy as np
import pandas as pd
from time                             import time
from pymatgen                         import periodic_table, Composition
from sklearn.model_selection          import GridSearchCV, KFold
from sklearn.neighbors                import KNeighborsRegressor
from sklearn.preprocessing            import StandardScaler
from sklearn.model_selection          import train_test_split
from sklearn.neighbors                import NearestNeighbors
from sklearn.metrics                  import mean_absolute_error
from sklearn.metrics                  import mean_squared_error
from sklearn.metrics                  import r2_score
from sklearn.model_selection          import cross_val_predict

start = time()
#%%
# function

def get_parameters(formula):
    """
    make parameters from chemical formula
    
    Parameters
    ----------
    formula : string
        chemical formula

    Returns
    -------
    array-like, shape = [2*numbers of atom]
        atomic number Z, numbers of atom
    """
    material = Composition(formula)
    features = []
    atomicNo = []
    natom = []
    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))
    features.extend(atomicNo)
    features.extend(natom)
    return features

def read_fxy_csv(name): 
    """
    read chemical formula, X, y from csv file
    
    Parameters
    ----------
    name : string
        csv file name

    Returns
    -------
    f : array-like, shape = [n_samples]
        chemical formulas
    X : array-like, shape = [n_samples, n_features]
        input parameters
    y : array-like, shape = [n_samples]
        output parameters
        
    """
    data = np.array(pd.read_csv(filepath_or_buffer=name, index_col=0,
                                header=0, sep=','))[:,:]
    f = np.array(data[:,0],dtype=np.unicode)
    y = np.array(data[:,1],dtype=np.float)
    X = np.array(data[:,2:],dtype=np.float)
    return f, X, y


def print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv):
    """
    print score of results of GridSearchCV (regression)

    Parameters
    ----------
    gscv : 
        GridSearchCV (scikit-learn)

    X_train : array-like, shape = [n_samples, n_features]
        X training data

    y_train : array-like, shape = [n_samples]
        y training data

    X_test : array-like, sparse matrix, shape = [n_samples, n_features]
        X test data

    y_test : array-like, shape = [n_samples]
        y test data

    cv : int, cross-validation generator or an iterable
        ex: 3, 5, KFold(n_splits=5, shuffle=True)

    Returns
    -------
    None
    """
    print()
    print("Best parameters set found on development set:")
    print(gscv.best_params_)
    y_calc = gscv.predict(X_train)
    rmse  = np.sqrt(mean_squared_error (y_train, y_calc))
    mae   =         mean_absolute_error(y_train, y_calc)
    r2    =         r2_score           (y_train, y_calc)
    print('C:  RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f}'\
    .format(rmse, mae, r2))

    y_incv = cross_val_predict(gscv, X_train, y_train, cv=cv)
    rmse  = np.sqrt(mean_squared_error (y_train, y_incv))
    mae   =         mean_absolute_error(y_train, y_incv)
    r2    =         r2_score           (y_train, y_incv)
    print('CV: RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f}'\
    .format(rmse, mae, r2))

    y_pred = gscv.predict(X_test)
    rmse  = np.sqrt(mean_squared_error (y_test, y_pred))
    mae   =         mean_absolute_error(y_test, y_pred)
    r2    =         r2_score           (y_test, y_pred)
    print('TST:RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f}'\
    .format(rmse, mae, r2))
    print()


def ad_knn(X_train, X_test):
    """
    Determination of Applicability Domain (k-Nearest Neighbor)
    
    Parameters
    ----------
    X_train : array-like, shape = [n_samples, n_features]
        X training data

    X_test : array-like, shape = [n_samples, n_features]
        X test data

    Returns
    -------
    array-like, shape = [n_samples]
        -1 (outer of AD) or 1 (inner of AD)
    """
    n_neighbors = 5      # number of neighbors
    r_ad = 0.9           # ratio of X_train inside AD / all X_train
    # ver.1
    neigh = NearestNeighbors(n_neighbors=n_neighbors+1)
    neigh.fit(X_train)
    dist_list = np.mean(neigh.kneighbors(X_train)[0][:,1:], axis=1)
    dist_list.sort()
    ad_thr = dist_list[round(X_train.shape[0] * r_ad) - 1]
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X_train)
    dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
    y_appd = 2 * (dist < ad_thr) -1

    return y_appd



#%%
# make tc_data.csv & tc_pred.csv
print()
print('make tc_data.csv & tc_pred.csv')
print()
df = pd.read_csv(filepath_or_buffer='tc.csv',
                 header=0, sep=',', usecols=[0, 2, 6])
df['Tc'] = df['     Tc [K]'].apply(float)
df['P'] = df['  P [GPa]'].apply(float)
df['list'] = df['formula'].apply(get_parameters)
df['formula'] = df['formula'].apply(lambda x: x.strip())
for i in range(len(get_parameters('H3S'))):
    name = 'prm' + str(i)
    df[name] = df['list'].apply(lambda x: x[i])
df = df.drop(['     Tc [K]', '  P [GPa]', 'list'], axis=1)
df.to_csv("tc_data.csv")

tc = 0.0
yx = []
zatom2 = 1
atom2 = periodic_table.get_el_sp(zatom2)
for zatom1 in range(3,86):
    atom1 = periodic_table.get_el_sp(zatom1)
    if(not atom1.is_noble_gas):
        for natom1 in range(1,2):
            for natom2 in range(1,11):
                for ip in range(50,550,50):
                    str_mat=str(atom1)+str(natom1)+str(atom2)+str(natom2)
                    material = Composition(str_mat)
                    temp = [material.reduced_formula, tc, float(ip)]
                    temp.extend(get_parameters(material.reduced_formula))
                    yx.append(temp[:])

properties=df.columns.values
df_test = pd.DataFrame(yx, columns=properties)
df_test = df_test.drop_duplicates()
df_test.to_csv("tc_pred.csv")

print('{:.2f} seconds '.format(time() - start))

#%%
range_n = np.arange( 1,  11, dtype=int)

nprm = len(get_parameters('H3S'))

name = 'kNN'
model = KNeighborsRegressor()
param_grid = {'n_neighbors':range_n}
output = 'tc_' + name + '.csv'

print()
print('read train & pred data from csv file')
print()
data_file = 'tc_data.csv'
_, X, y = read_fxy_csv(data_file)
pred_file = 'tc_pred.csv'
f_pred, X_pred, _ = read_fxy_csv(pred_file)

scaler = StandardScaler()

X = scaler.fit_transform(X)
P_pred = X_pred[:,0]
X_pred = scaler.transform(X_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%
# Set the parameters by cross-validation

n_splits = 3 
cv = KFold(n_splits=n_splits, shuffle=True)

gscv = GridSearchCV(model, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

#%%
# Re-learning with all data & best parameters -> Prediction
best = gscv.best_estimator_.fit(X, y)
y_pred = best.predict(X_pred)

#%%
# Applicability Domain (inside: +1, outside: -1)

y_appd = ad_knn(X, X_pred)

data = []
for i in range(len(X_pred)):
    temp = (f_pred[i], int(P_pred[i]), int(y_pred[i]), y_appd[i])
    data.append(temp)

properties=['formula','P', 'Tc', 'AD']
df = pd.DataFrame(data, columns=properties)
df.sort_values('Tc', ascending=False, inplace=True)

df.to_csv(output, index=False)
print('Predicted Tc is written in file {}'.format(output))

print('{:.2f} seconds '.format(time() - start))
