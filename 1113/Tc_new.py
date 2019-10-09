# -*- coding: utf-8 -*-
"""
A.Make Tc csv files
1.convert tc_new.csv -> tc_new_train.csv
    chemical formula -> parameters(atomic No. & number of atoms)
    ex) S2H5  -> 16.0, 1.0, 2.0, 5.0
2.make tc_new_test.csv
   all A_n H_m (n,m=1,...,10)

B.Hydride Tc Regression
1.Hydride Tc Regression (RR, LASSO, EN, k-NN, RF, SVM)
2.Applicability Domain (k-NN)
3.Double Cross Validation
4.y-randamization

Parameters
----------
Nothing

Returns
-------
Nothing

Input file
----------
tc_new.csv:

Temporary file
----------
tc_new_train.csv:
    Tc, atomic number 1&2, the number of atoms 1&2, pressure 
    of already calculated materials

tc_new_test.csv:
    Tc, atomic number 1&2, the number of atoms 1&2, pressure 
    of XnHm (n,m=1,...,10): X=He~At (without rare gas)

Outnput file
------------
Tc_'model_name'_AD_DCV.csv:
    chemical formula, P, Tc, AD

-----------------------------------
Created on Tue Nov 13 13:18:00 2018

@author: Akitaka
"""

import numpy as np
import pandas as pd
from pymatgen import Composition
from pymatgen import periodic_table 


# A.Make Tc csv files
def get_parameters(formula):
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

lcsv=False
if(lcsv):
    df = pd.read_csv(filepath_or_buffer='tc_new.csv',
                     header=0, sep=',', usecols=[0, 2, 6])
    print(df.head())
    print(df.index)
    print(df.columns)
    
    df['Tc'] = df['     Tc [K]'].apply(float)
    df['P'] = df['  P [GPa]'].apply(float)
    df['list'] = df['formula'].apply(get_parameters)
    df['formula'] = df['formula'].apply(lambda x: x.strip())
    for i in range(4):
        name = 'prm' + str(i)
        df[name] = df['list'].apply(lambda x: x[i])
    df = df.drop(['     Tc [K]', '  P [GPa]', 'list'], axis=1)
    df.to_csv("tc_new_train.csv")
    print(df.head(5))
    
    tc = 0.0
    yx = []
    zatom2 = 1
    atom2 = periodic_table.get_el_sp(zatom2)
    for zatom1 in range(3,86):
        atom1 = periodic_table.get_el_sp(zatom1)
        if(not atom1.is_noble_gas):
            for natom1 in range(1,11):
                for natom2 in range(1,11):
                    for ip in range( 50,550,50):
                        str_mat=str(atom1)+str(natom1)+str(atom2)+str(natom2)
                        material = Composition(str_mat)
                        temp = [material.reduced_formula, tc, float(ip)]
                        temp.extend(get_parameters(material.reduced_formula))
                        yx.append(temp[:])
    
    properties=df.columns.values
    df_test = pd.DataFrame(yx, columns=properties)
    print(df_test.duplicated().value_counts())
    df_test = df_test.drop_duplicates()
    df_test.to_csv("tc_new_test.csv")
    print(df_test.head(5))



#%%
import pandas as pd
from time                    import time
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.svm             import SVR
from sklearn.linear_model    import Ridge, Lasso, ElasticNet
from pymatgen                import periodic_table, Composition
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from my_library              import print_gscv_score_rgr, dcv_rgr
from my_library              import y_randamization_rgr, ad_knn, ad_ocsvm

def read_fxy_csv(name): 
    data = np.array(pd.read_csv(filepath_or_buffer=name, index_col=0,
                                header=0, sep=','))[:,:]
    f = np.array(data[:,0],dtype=np.unicode)
    y = np.array(data[:,1],dtype=np.float)
    X = np.array(data[:,2:],dtype=np.float)
    return f, X, y

start = time()

range_f =  0.1*np.arange(  1, 11, dtype=int)
range_n = np.arange( 1,  11, dtype=int)
range_c = 2**np.arange(  -5+15, 11, dtype=float)
range_e = 2**np.arange( -10+5,  1, dtype=float)
range_g = 2**np.arange( -20+30, 11, dtype=float)

test = {
'RR'   :{'name':'Ridge Regression',
         'model':Ridge(),
         'param':{'alpha':range_f}},
'EN'   :{'name':'Elastic Net     ',
         'model':ElasticNet(),
         'param':{'alpha':range_f, 'l1_ratio':range_f}},
'LASSO':{'name':'LASSO           ',
         'model':Lasso(),
         'param':{'alpha':range_f}},
'kNN'  :{'name':'kNN             ',
         'model':KNeighborsRegressor(),
         'param':{'n_neighbors':range_n}},
'RF'   :{'name':'Random Forest',
         'model':RandomForestRegressor(),
         'param':{'max_features':range_f}},
'SVR'  :{'name':'SVR',
         'model':SVR(),
         'param':{'gamma': range_g, 
                  'C': range_c,
                  'epsilon': range_e}},
}


key = 'kNN' # 'RR' 'EN', 'LASSO', 'kNN', 'RF', 'SVR'
name = test[key]['name']
model = test[key]['model']
param_grid = test[key]['param']
output = 'Tc_new_' + key + '_AD_DCV.csv'
print(name)
print(model)
print(param_grid)
print(output)

print()
print('read train & test data from csv file')
print()
train_file = 'tc_new_train.csv'
f_train, X_train, y_train = read_fxy_csv(train_file)
test_file = 'tc_new_test.csv'
f_test, X_test, y_test = read_fxy_csv(test_file)


iscaler=1
if(iscaler==1):
    scaler = StandardScaler()
else:
    scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
# print statistics of database
if(False):
    data = pd.read_csv(filepath_or_buffer='tc_new_train.csv',
                       index_col=0, header=0, sep=',')
    data.drop(['formula'], axis=1, inplace=True)
    print(data.describe())

#%%
# Set the parameters by cross-validation

n_splits = 5 
cv = ShuffleSplit(n_splits=n_splits, test_size=0.2)
cv = KFold(n_splits=n_splits, shuffle=True)
gscv = GridSearchCV(model, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

#%%
# Prediction
y_pred = gscv.predict(X_test)

# Applicability Domain (inside: +1, outside: -1)
iappd = 1
if(iappd == 1):
    y_appd = ad_knn(X_train, X_test)
else:
    y_appd = ad_ocsvm(X_train, X_test)

data = []
for i in range(len(X_test)):
    temp = (f_test[i], int(X_test[i][0]), int(y_pred[i]), y_appd[i])
    data.append(temp)

properties=['formula','P', 'Tc', 'AD']
df = pd.DataFrame(data, columns=properties)
df.sort_values('Tc', ascending=False, inplace=True)

# df.to_csv(output, index=False)
df_in_ = df[df.AD ==  1]
df_in_.to_csv(output, index=False)
print('Predicted Tc is written in file {}'.format(output))

#%%
niter=10
if(True):
    dcv_rgr(X_train, y_train, model, param_grid, niter)
    y_randamization_rgr(X_train, y_train, model, param_grid, niter)

# print(X_train[:10])
print('{:.2f} seconds '.format(time() - start))
