# -*- coding: utf-8 -*-
"""
Machine learning energy gap of ABO2 (regression)
Created on Fri Jun 22 11:12:16 2018

@author: Akitaka
"""
print(__doc__)

# modules
# {{{
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# }}}
#
# functions printing score
#
def print_score(y_test,y_pred): #{{{
    rmse  = np.sqrt(mean_squared_error (y_test,y_pred))
    mae   =         mean_absolute_error(y_test,y_pred)
    r2    =         r2_score           (y_test,y_pred)
    if(mae > 0):
        rmae = np.sqrt(mean_squared_error (y_test,y_pred))/mae
    else:
        rmae = 0.0
#    print('RMSE, MAE, RMSE/MAE, R^2 = {:.3f}, {:.3f}, {:.3f}, {:.3f}'\
#    .format(rmse, mae, rmae, r2))
    print(' {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(rmse, mae, rmae, r2))

#}}}
def print_search_score(grid_search,X_train,y_train): #{{{
    print('search score')
    scores = cross_val_score(grid_search.best_estimator_,X_train,y_train,cv=5)
    np.set_printoptions(precision=3)
    print('In cross validation with best params = ',grid_search.best_params_)
    print('  scores for each fold = ' , scores)
    print('  ave, std = {:.3f} (+/-{:.3f})'\
    .format(scores.mean(), scores.std()*2 ))
    print('  ave = grid_search.best_score  :', grid_search.best_score_)
#   print('')
#   y_pred = grid_search.predict(X_train)
#   print('r2_score(y_train,y_pred ) : %.3f' % (r2_score(y_train,y_pred )))
#   print('= grid_search.score : %.3f' % (grid_search.score(X_train,y_train)))
#   print('')
#   print('grid_search.cv_results_ = ',grid_search.cv_results_)
    print('')
    i_best=grid_search.cv_results_['params'].index(grid_search.best_params_)
    score0=grid_search.cv_results_['split0_test_score'][i_best]
    score1=grid_search.cv_results_['split1_test_score'][i_best]
    score2=grid_search.cv_results_['split2_test_score'][i_best]
    score3=grid_search.cv_results_['split3_test_score'][i_best]
    score4=grid_search.cv_results_['split4_test_score'][i_best]
    print('grid_search.cv_results_[params] = ',i_best)
    print('grid_search.cv_results_[split0_test_score] = {:6.3f}'.format(score0))
    print('grid_search.cv_results_[split1_test_score] = {:6.3f}'.format(score1))
    print('grid_search.cv_results_[split2_test_score] = {:6.3f}'.format(score2))
    print('grid_search.cv_results_[split3_test_score] = {:6.3f}'.format(score3))
    print('grid_search.cv_results_[split4_test_score] = {:6.3f}'.format(score4))
    print('  ave = {:.3f}'.format((score0+score1+score2+score3+score4)/5.0))
#   print('')
#   print('grid_search.cv_results_[mean_test_score] = ',grid_search.cv_results_['mean_test_score'])
#   print('')
#   means = grid_search.cv_results_['mean_test_score']
#   stds  = grid_search.cv_results_['std_test_score']
#   for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#       print('%0.3f (+/-%0.03f) for %r' % (mean, std*2, params))
#   or 
#   print('grid_scores :', grid_search.grid_scores_) 
#   NOTE: Don't use grid_scores_
#   The grid_scores_ attribute was deprecated in version 0.18
#    in favor of the more elaborate cv_results_ attribute.
#   The grid_scores_ attribute will not be available from 0.20.
#}}}

# read data from csv file
name = 'test1.csv'
data = np.array(pd.read_csv(name))[:,:]
y=data[:,8]
X=data[:,0:2]
#print(y[:5])
#print(X[:5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                       test_size=0.3, random_state=0)
X_train = X[:]
X_test  = X[:]
y_train = y[:]
y_test  = y[:]


from sklearn.model_selection          import ShuffleSplit
from sklearn.svm                      import SVR
from sklearn.linear_model             import LinearRegression
from sklearn.linear_model             import OrthogonalMatchingPursuit
from sklearn.linear_model             import RANSACRegressor
from sklearn.linear_model             import TheilSenRegressor
from sklearn.linear_model             import BayesianRidge
from sklearn.linear_model             import Lasso
from sklearn.linear_model             import ElasticNet
from sklearn.linear_model             import Ridge
from sklearn.tree                     import DecisionTreeRegressor
from sklearn.ensemble                 import RandomForestRegressor
from sklearn.ensemble                 import RandomTreesEmbedding
from sklearn.neural_network           import MLPRegressor
from sklearn.mixture                  import BayesianGaussianMixture
from sklearn.neighbors                import KNeighborsRegressor
from sklearn.neighbors                import RadiusNeighborsRegressor
from sklearn.cross_decomposition      import PLSRegression
from sklearn.naive_bayes              import GaussianNB
from sklearn.naive_bayes              import MultinomialNB
from sklearn.gaussian_process         import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

# sklearn NO random forest KAIKI
lr  = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
rte = RandomTreesEmbedding()
mr  = MLPRegressor(max_iter=1000)
omp = OrthogonalMatchingPursuit()
ran = RANSACRegressor()
tsr = TheilSenRegressor(random_state=42)
br  = BayesianRidge(n_iter=300,tol=0.001)
bgm = BayesianGaussianMixture()
knr = KNeighborsRegressor(n_neighbors=5)
rnr = RadiusNeighborsRegressor(radius=1.0)
pls = PLSRegression(n_components=1)
gnb = GaussianNB()
mnb = MultinomialNB()
svl = SVR(kernel='linear') 
svr = SVR() 
las = Lasso() 
en  = ElasticNet()
rr  = Ridge()
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)


estimators = {'LR ':lr , 'DTR':dtr, 'RFR':rfr, 'OMP':omp, 'RAN':ran, 'BR ':br, 
              'BGM':bgm, 'KNR':knr, 'RNR':rnr, 'PLS':pls, 'SVL':svl, 'SVR':svr,
              'LAS':las, 'EN ':en,  'RR ':rr,  'GPR':gpr, 'TSR':tsr}

lgraph = False
# KOUSA KENSHO SIMASU
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
for k,v in estimators.items():
    start = time()

    print(k, end="")
    # step 1. model
    mod = v

    # step 2. learning -> score
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_train)
    if(lgraph):
        lw = 2
        plt.scatter(X_train[:,1], y_train, color='darkorange', label='data')
        plt.plot(X_train[:,1], y_pred, color='navy', lw=lw, label='model')
        plt.xlabel('NB of ABO2')
        plt.ylabel('Eg')
        plt.title(k)
        plt.legend()
        plt.show()
    
#    print('learning   score: ',end="")
    print_score(y_train, y_pred)

    # step 3. predict
    y_pred = mod.predict(X_test)
    # replace NaN = -1
    y_pred[np.isnan(y_pred)]= -1.0

    # step 4. score
#    print('prediction score: ',end="")
#    print_score(y_test,  y_pred)
#    print('{:.2f} seconds '.format(time() - start))    