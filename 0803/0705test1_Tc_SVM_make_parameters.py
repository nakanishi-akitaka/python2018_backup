# -*- coding: utf-8 -*-
"""
Calculate parameters from physical properties

Created on Thu Jul  5 13:27:09 2018

@author: Akitaka
Ref:
"A Data-Driven Statistical Model for Predicting the Critical Temperature
 of a Superconductor"
https://arxiv.org/abs/1803.10260

ref:
20180315test0.py
20180521test2.py
20180426test2.py

1.Atomic Mass atomic, mass units (AMU)
2.First Ionization Energy, kilo-Joules per mole (kJ/mol)
3.Atomic Radius, picometer (pm)
4.Density, kilograms per meters cubed (kg/m3)
5.Electron Affinity, kilo-Joules per mole (kJ/mol)
6.Fusion Heat, kilo-Joules per mole (kJ/mol)
7.Thermal Conductivity, watts per meter-Kelvin (W/(m × K))
8.Valence no units typical number of chemical bonds formed by the element
Table 1: This table shows the properties of an element which are used for creating features to
predict Tc.

ref of unit
https://mendeleev.readthedocs.io/en/stable/data.html
https://mendeleev.readthedocs.io/en/stable/_modules/mendeleev/tables.html
https://mendeleev.readthedocs.io/en/stable/code.html

Mean = µ = (t1 + t2)/2 
Weighted mean = ν = (p1t1) + (p2t2) 
Geometric mean = (t1t2)**1/2
Weighted geometric mean = (t1)**p1*(t2)**p2
Entropy = −w1 ln(w1) − w2 ln(w2)
Weighted entropy = −A ln(A) − B ln(B) 
Range = t1 − t2 (t1 > t2)
Weighted range = p1 t1 − p2 t2
Standard deviation = [(1/2)((t1 − µ)**2 + (t2 − µ)**2)]**1/2
Weighted standard deviation = [p1(t1 − ν)**2 + p2(t2 − ν)**2)]**1/2
Table 2:
"""

import numpy as np
from pymatgen import Composition
from mendeleev import element

def calc_parameters(n1,n2,t1,t2):
    p1 = n1/(n1+n2)
    p2 = n2/(n1+n2)
    w1 = t1/(t1+t2)
    w2 = t2/(t1+t2)
    A  = (p1*w1)/(p1*w1+p2*w2)
    B  = (p2*w2)/(p1*w1+p2*w2)
    mu = (t1 + t2)/2
    nu = p1 * t1 + p2 * t2
    gm = np.sqrt(t1*t2)
    wg = t1**p1*t2**p2 
    en = -w1*np.log(w1) - w2*np.log(w2)
    we = - A*np.log(A)  -  B*np.log(B)
    ra = np.abs(t1-t2)
    wr = np.abs(p1*t1-p2*t2)
    sd = np.sqrt(((t1-mu)**2+(t2-mu)**2)/2)
    ws = np.sqrt(p1*(t1-nu)**2+p2*(t2-nu)**2)
    return mu, nu, gm, wg, en, we, ra, wr, sd, ws
#    Mean = µ = (t1 + t2)/2 
#    Weighted mean = ν = (p1t1) + (p2t2) 
#    Geometric mean = (t1t2)**1/2
#    Weighted geometric mean = (t1)**p1*(t2)**p2
#    Entropy = −w1 ln(w1) − w2 ln(w2)
#    Weighted entropy = −A ln(A) − B ln(B) 
#    Range = t1 − t2 (t1 > t2)
#    Weighted range = p1 t1 − p2 t2
#    Standard deviation = [(1/2)((t1 − µ)**2 + (t2 − µ)**2)]**1/2
#    Weighted standard deviation = [p1(t1 − ν)**2 + p2(t2 − ν)**2)]**1/2
#    
#    print(dir(Element(x)))

# read datasets from csv file
# ref: 0627test10.py

import pandas as pd

print()
print('make parameters from csv file')
print()

file = "../python_work_fs01/2018/0330/bandgapDFT.csv"
data = np.array(pd.read_csv(file,header=None))[:,:]
# print(data)
y=data[:,1]
X=data[:,0]

Xconv = []
yconv = []
Xy = []
from time import time
start = time()
for i in range(len(X)):
    material = Composition(X[i])
    natom = []
    amass = []
    eion1 = []
    aradi = []
    dense = []
    elaff = []
    meltp = []
    therm = []
    nvale = []
    lcalc = True
    for x in material:
        natom.append(material.get_atomic_fraction(x)*material.num_atoms)
        x = element(str(x))
        if(x.electron_affinity==None):
            lcalc = False
        else:
            x.electron_affinity += 2.5
        if(x.thermal_conductivity==None):
            lcalc = False
        if(x.density==None):
            lcalc = False
        if(lcalc):
            amass.append(x.mass) # AMU
            eion1.append(x.ionenergies[1]) # a dictionary with ionization energies in eV
            aradi.append(x.atomic_radius_rahm) # Atomic radius by Rahm et al.	pm
            dense.append(x.density) # g/cm3
            elaff.append(x.electron_affinity) # eV
            meltp.append(x.melting_point) # Kelvin
            therm.append(x.thermal_conductivity) # W /(m K) 
            nvale.append(x.nvalence()) # no unit
    if(lcalc): 
        allprm = []
        pplist = [amass, eion1, aradi, dense, elaff, meltp, therm, nvale]
        for pp in pplist:
            params = calc_parameters(*natom, *pp)
            allprm += list(params)    
        Xconv.append(allprm)
        yconv.append(y[i])
        allprm.append(y[i])
        Xy.append(allprm)

# ref: 0622/test1.py
test = pd.DataFrame(Xy)
test.to_csv('bandgapDFT_conv.csv')
print('{:.2f} seconds '.format(time() - start))
#%%
from time import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

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
    print("Grid scores on development set:")
    print()
    means = gscv.cv_results_['mean_test_score']
    stds = gscv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
        print("{:.3f} (+/-{:.03f}) for {:}".format(mean, std * 2, params))
       
# training data: y = sin(x) + noise
#
X_train = np.array(Xconv)
y_train = np.array(yconv)
X_test = np.array(Xconv)
y_test = np.array(yconv)

start = time()
print('')
print('')
print('# 1. SVR with default hyper parameters')

# step 1. model
mod = SVR() 

# step 2. learning with optimized parameters
# search range
#range_c = 2**np.arange( -5, 10, dtype=float)
#range_g = 2**np.arange( -20, 10, dtype=float)
range_c = 2**np.arange( -5, 0, dtype=float)
range_g = 2**np.arange( -5, 0, dtype=float)

# Set the parameters by cross-validation
param_grid = [
        {'kernel': ['rbf'], 'gamma': range_g,'C': range_c},
        {'kernel': ['linear'], 'C': range_c}
        ]

# grid_search = GridSearchCV(mod, param_grid, cv=5, n_jobs=-1)
gscv = GridSearchCV(mod, param_grid, cv=5)
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
plt.scatter(X_test[:,80], y_test,  color='black', label='test data')
plt.plot(X_test[:,80], y_pred, color='blue', linewidth=3, label='model')
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()

