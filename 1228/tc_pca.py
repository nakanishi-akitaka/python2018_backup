# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:06:39 2018

@author: Akitaka
"""

import numpy as np
import pandas as pd
from time                             import time
from pymatgen                         import Composition
from sklearn.preprocessing            import StandardScaler
from sklearn.neighbors                import NearestNeighbors
from sklearn.metrics                  import mean_absolute_error
from sklearn.metrics                  import mean_squared_error
from sklearn.metrics                  import r2_score
from sklearn.model_selection          import cross_val_predict

start = time()

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



# make tc_data.csv & tc_pred.csv
print()
print('make tc_temp.csv')
print()
df = pd.read_csv(filepath_or_buffer='tc.csv',
                 header=0, sep=',', usecols=[0,2,6])
df['Tc'] = df['     Tc [K]'].apply(float)
df['P'] = df['  P [GPa]'].apply(float)
df['list'] = df['formula'].apply(get_parameters)
df['formula'] = df['formula'].apply(lambda x: x.strip())
for i in range(len(get_parameters('H3S'))):
    name = 'prm' + str(i)
    df[name] = df['list'].apply(lambda x: x[i])
df = df.drop(['     Tc [K]', '  P [GPa]', 'list'], axis=1)
df.to_csv("tc_temp.csv")

print('{:.2f} seconds '.format(time() - start))

_, X, y = read_fxy_csv('tc_temp.csv')

from sklearn.decomposition import PCA 

scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

pca = PCA(n_components=5)
X_transformed = pca.fit_transform(X)
print("Percentage of variance explained by each of the selected components.")
print(pca.explained_variance_ratio_)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.subplot(2, 1, 1)
plt.scatter(X_transformed[:,0],X_transformed[:,1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show