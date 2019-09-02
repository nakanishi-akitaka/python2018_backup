# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:41:37 2018

@author: Akitaka
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics         import mean_absolute_error
from sklearn.metrics         import mean_squared_error
from sklearn.metrics         import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict

def print_gscv_score(gscv):
    print("Best parameters set found on development set:")
    print()
    print(gscv.best_params_)
    print()
    print("Grid scores on development set:")
    print()
#    means = gscv.cv_results_['mean_test_score']
#    stds = gscv.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
#        print("{:.3f} (+/-{:.03f}) for {:}".format(mean, std * 2, params))

def print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv):
    print()
    print("Best parameters set found on development set:")
    print(gscv.best_params_)
    y_calc = gscv.predict(X_train)
    rmse  = np.sqrt(mean_squared_error (y_train, y_calc))
    mae   =         mean_absolute_error(y_train, y_calc)
    r2    =         r2_score           (y_train, y_calc)
    print('RMSEc,  MAEc,  Rc^2  = {:.3f}, {:.3f}, {:.3f}'\
    .format(rmse, mae, r2))
    yyplot(y_train, y_calc)

    y_incv = cross_val_predict(gscv, X_train, y_train, cv=cv)
    rmse  = np.sqrt(mean_squared_error (y_train, y_incv))
    mae   =         mean_absolute_error(y_train, y_incv)
    r2    =         r2_score           (y_train, y_incv)
    print('RMSEcv, MAEcv, Rcv^2 = {:.3f}, {:.3f}, {:.3f}'\
    .format(rmse, mae, r2))
    yyplot(y_train, y_incv)

    y_pred = gscv.predict(X_test)
    rmse  = np.sqrt(mean_squared_error (y_test, y_pred))
    mae   =         mean_absolute_error(y_test, y_pred)
    r2    =         r2_score           (y_test, y_pred)
    print('RMSEp,  MAEp,  Rp^2  = {:.3f}, {:.3f}, {:.3f}'\
    .format(rmse, mae, r2))
    yyplot(y_test, y_pred)
    print()


def print_score_rgr(y_test,y_pred):
    rmse  = np.sqrt(mean_squared_error (y_test,y_pred))
    mae   =         mean_absolute_error(y_test,y_pred)
    if(mae > 0):
        rmae = np.sqrt(mean_squared_error (y_test,y_pred))/mae
    else:
        rmae = 0.0
    r2    =         r2_score           (y_test,y_pred)
    print('RMSE, MAE, RMSE/MAE, R^2 = {:.3f}, {:.3f}, {:.3f}, {:.3f}'\
    .format(rmse, mae, rmae, r2))
    if(rmae > np.sqrt(np.pi/2.0)):
        print("RMSE/MAE = {:.3f} > sqrt(pi/2), some sample have large error?"\
              .format(rmae))
    elif(rmae < np.sqrt(np.pi/2.0)):
        print("RMSE/MAE = {:.3f} < sqrt(pi/2), each sample have same error?"\
              .format(rmae))
    elif(rmae == np.sqrt(np.pi/2.0)):
        print("RMSE/MAE = {:.3f} = sqrt(pi/2), normal distribution error?"\
              .format(rmae))


def yyplot(y_obs, y_pred):
    fig = plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.title("yy-plot")
    plt.scatter(y_obs, y_pred)
    y_all = np.concatenate([y_obs, y_pred])
    ylowlim = np.amin(y_all) - 0.05 * np.ptp(y_all)
    yupplim = np.amax(y_all) + 0.05 * np.ptp(y_all)
    plt.plot([ylowlim, yupplim],
             [ylowlim, yupplim],'k-')
    plt.ylim( ylowlim, yupplim)
    plt.xlim( ylowlim, yupplim)
    plt.xlabel("y_observed")
    plt.ylabel("y_predicted")
    
    plt.subplot(1,2,2)
    error = np.array(y_pred-y_obs)
    plt.hist(error)
    plt.title("Error histogram")
    plt.xlabel('prediction error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    return fig

def dcv(X,y,mod,param_grid):
    # parameters
    ns_in = 3 # n_splits for inner loop
    ns_ou = 3 # n_splits for outer loop
    
    i = 1 # index of loop
    scores = np.array([]) # list of test scores in outer loop
    kf_ou = KFold(n_splits=ns_ou, shuffle=True)
    
    # [start] outer loop for test of the generalization error
    for train_index, test_index in kf_ou.split(X):
        X_train, X_test = X[train_index], X[test_index] # inner loop CV
        y_train, y_test = y[train_index], y[test_index] # outer loop 
    
        # [start] inner loop CV for hyper parameter optimization
        kf_in = KFold(n_splits=ns_in, shuffle=True)
        gscv = GridSearchCV(mod, param_grid, cv=kf_in)
        gscv.fit(X_train, y_train)
        # [end] inner loop CV for hyper parameter optimization
        
        # test of the generalization error
        score = gscv.score(X_test, y_test)
        scores = np.append(scores, score)
#        print('dataset: {}/{}  accuracy of inner CV: {:.3f} time: {:.3f} s'.\
#              format(i,ns_ou,score,(time() - start)))
        i+=1
    
    # [end] outer loop for test of the generalization error
    print('  ave, std of accuracy of inner CV: {:.3f} (+/-{:.3f})'\
        .format(scores.mean(), scores.std()*2 ))




def optimize_gamma(X, gammas):
    # Optimize gamma by maximizing variance in Gram matrix
    var_matrix = list()
    for gamma in gammas:
        gram_matrix = np.exp(-gamma*((X[:, np.newaxis] - X)**2).sum(axis=2))
        var_matrix.append(gram_matrix.var(ddof=1))
    return gammas[ np.where( var_matrix == np.max(var_matrix) )[0][0] ]
