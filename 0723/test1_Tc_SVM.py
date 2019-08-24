# -*- coding: utf-8 -*-
"""
ML of Tc by SVM

Created on Mon Jul 23 15:25:22 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
from matplotlib              import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.svm             import SVR
from pymatgen                import Element
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from my_library              import print_gscv_score
from my_library              import print_score

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

def read_xy_csv(name): 
    data = np.array(pd.read_csv(filepath_or_buffer=name,
                               header=None,sep=','))[:,:]
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
    data = pd.read_csv(filepath_or_buffer='tc_train.csv',header=None, sep=',')
    data.columns = ['tc','z1', 'z2', 'n1','n2','p']
    data.drop('z2', axis=1, inplace=True)
    print(data.describe())

ltest=True
if(ltest):
    # range_c = 2**np.arange( -5,  10, dtype=float)
    # range_e = 2**np.arange( -10,  0, dtype=float)
    # range_g = 2**np.arange( -20, 10, dtype=float)
    range_c = 2**np.arange(  10, 11, dtype=float)
    range_e = 2**np.arange(  -1,  1, dtype=float)
    range_g = 2**np.arange(  10, 11, dtype=float)
    
    print()
    print('Search range')
    print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
    print('e = ', range_e[0], ' ... ',range_e[len(range_e)-1])
    print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
    print()
    
    # Set the parameters by cross-validation
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
    ])
    
    param_grid = [
    {'svr__kernel': ['rbf'], 'svr__gamma': range_g,
     'svr__C': range_c,'svr__epsilon': range_e},
    ]
    
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    
    score='neg_mean_absolute_error'
    
    gscv = GridSearchCV(pipe, param_grid, cv=cv, scoring=score)
    gscv.fit(X_train, y_train)
    print_gscv_score(gscv)
    
    y_pred = gscv.predict(X_train)
    print('train data: ',end="")
    print_score(y_train, y_pred)
    
    
    # yy-plot (train)
    y_pred = gscv.predict(X_train)
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
    # ref:
    # http://univprof.com/archives/16-07-20-4857140.html
    plt.subplot(1,2,2)
    y_pred = gscv.predict(X_train)
    error = np.array(y_pred-y_train)
    plt.hist(error)
    plt.title("Gaussian? (train)")
    plt.xlabel('prediction error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    # step 3. predict
    y_pred = gscv.predict(X_test)
    print_high_Tc(X_test,y_pred)
    
    print('{:.2f} seconds '.format(time() - start))
    
