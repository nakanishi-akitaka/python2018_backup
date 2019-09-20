# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:29:27 2018

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
from sklearn.metrics         import confusion_matrix, accuracy_score
from sklearn.neighbors       import NearestNeighbors
from sklearn.svm             import OneClassSVM

def print_gscv_score(gscv):
    """
    print score of results of GridSearchCV

    Parameters
    ----------
    gscv :
        GridSearchCV (scikit-learn)

    Returns
    -------
    None
    """
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
    lgraph = False
    print()
    print("Best parameters set found on development set:")
    print(gscv.best_params_)
    y_calc = gscv.predict(X_train)
    rmse  = np.sqrt(mean_squared_error (y_train, y_calc))
    mae   =         mean_absolute_error(y_train, y_calc)
    r2    =         r2_score           (y_train, y_calc)
    print('C:  RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f}'\
    .format(rmse, mae, r2))
    if(lgraph):
        yyplot(y_train, y_calc)

    y_incv = cross_val_predict(gscv, X_train, y_train, cv=cv)
    rmse  = np.sqrt(mean_squared_error (y_train, y_incv))
    mae   =         mean_absolute_error(y_train, y_incv)
    r2    =         r2_score           (y_train, y_incv)
    print('CV: RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f}'\
    .format(rmse, mae, r2))
    if(lgraph):
        yyplot(y_train, y_incv)

    y_pred = gscv.predict(X_test)
    rmse  = np.sqrt(mean_squared_error (y_test, y_pred))
    mae   =         mean_absolute_error(y_test, y_pred)
    r2    =         r2_score           (y_test, y_pred)
    print('TST:RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f}'\
    .format(rmse, mae, r2))
    if(lgraph):
        yyplot(y_test, y_pred)

#    y_calc = gscv.predict(X_train)
#        gscv.fit(X_train, y_train, cv=3)
#       -> split X_train, y_train & optimize hyper parameters
#       -> finally, learn with all X_train, y_train 
#    C:  RMSE, MAE, R^2 = score for training data
#    CV: RMSE, MAE, R^2 = score for validation data
#              Validation data is not used, but CV is used. 
#    TST:RMSE, MAE, R^2 = score for test data
#    In dcv_rgr, 
#    DCV:RMSE, MAE, R^2 = average and standard deviation of score for test data
    print()




def print_gscv_score_clf(gscv, X_train, X_test, y_train, y_test, cv):
    """
    print score of results of GridSearchCV (classification)

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
    tn, fp, fn, tp = confusion_matrix(y_train, y_calc).ravel()
    print('C:  TP, FP, FN, TN, Acc. = {0}, {1}, {2}, {3}, {4:.3f}'.\
          format(tp, fp, fn, tn, accuracy_score(y_train, y_calc)))

    y_incv = cross_val_predict(gscv, X_train, y_train, cv=cv)
    tn, fp, fn, tp = confusion_matrix(y_train, y_incv).ravel()
    print('CV: TP, FP, FN, TN, Acc. = {0}, {1}, {2}, {3}, {4:.3f}'.\
          format(tp, fp, fn, tn, accuracy_score(y_train, y_incv)))

    y_pred = gscv.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('TST:TP, FP, FN, TN, Acc. = {0}, {1}, {2}, {3}, {4:.3f}'.\
          format(tp, fp, fn, tn, accuracy_score(y_test, y_pred)))
    print()




def print_score_rgr(y_test,y_pred):
    """
    print score of results of regression

    Parameters
    ----------
    y_test : array-like, shape = [n_samples]
        y test data

    y_pred : array-like, shape = [n_samples]
        y predicted data

    Returns
    -------
    None
    """
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
    """
    print yy-plot 

    Parameters
    ----------
    y_obs : array-like, shape = [n_samples]
        y observed data

    y_pred : array-like, shape = [n_samples]
        y predicted data

    Returns
    -------
    Figure object
    """
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
    """
    Double cross validation

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        X training+test data

    y : array-like, shape = [n_samples]
        y training+test data

    mod : 
        machine learning model (scikit-learn)

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored.

    Returns
    -------
    None
    """
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




def dcv_rgr(X, y, model, param_grid, niter):
    """
    Double cross validation (regression)

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        X training+test data

    y : array-like, shape = [n_samples]
        y training+test data

    model: 
        machine learning model (scikit-learn)

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored.

    niter : int
        number of DCV iteration

    Returns
    -------
    None
    """
    # parameters
    ns_in = 3 # n_splits for inner loop
    ns_ou = 3 # n_splits for outer loop
    scores = np.zeros((niter,3))
    for iiter in range(niter):
        ypreds = np.array([]) # list of predicted y in outer loop
        ytests = np.array([]) # list of y_test in outer loop
        kf_ou = KFold(n_splits=ns_ou, shuffle=True)
    
        # [start] outer loop for test of the generalization error
        for train_index, test_index in kf_ou.split(X):
            X_train, X_test = X[train_index], X[test_index] # inner loop CV
            y_train, y_test = y[train_index], y[test_index] # outer loop 
        
            # [start] inner loop CV for hyper parameter optimization
            kf_in = KFold(n_splits=ns_in, shuffle=True)
            gscv = GridSearchCV(model, param_grid, cv=kf_in)
            gscv.fit(X_train, y_train)
            # [end] inner loop CV for hyper parameter optimization
            
            # test of the generalization error
            ypred = gscv.predict(X_test)
            ypreds = np.append(ypreds, ypred)
            ytests = np.append(ytests, y_test)
        
        # [end] outer loop for test of the generalization error
        rmse  = np.sqrt(mean_squared_error (ytests, ypreds))
        mae   =         mean_absolute_error(ytests, ypreds)
        r2    =         r2_score           (ytests, ypreds)
#        print('DCV:RMSE, MAE, R^2 = {:.3f}, {:.3f}, {:.3f}'\
#        .format(rmse, mae, r2))
        scores[iiter,:] = np.array([rmse,mae,r2])

    means, stds = np.mean(scores, axis=0),np.std(scores, axis=0)
    print()
    print('Double Cross Validation')
    print('In {:} iterations, average +/- standard deviation'.format(niter))
#    print('RMSE: {:6.3f} (+/-{:6.3f})'.format(means[0], stds[0]))
#    print('MAE : {:6.3f} (+/-{:6.3f})'.format(means[1], stds[1]))
#    print('R^2 : {:6.3f} (+/-{:6.3f})'.format(means[2], stds[2]))
    print('DCV:RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f} (ave)'\
          .format(means[0], means[1], means[2]))
    print('DCV:RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f} (std)'\
          .format(stds[0], stds[1], stds[2]))



def dcv_clf(X, y, model, param_grid, niter):
    """
    Double cross validation (classification)

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        X training+test data

    y : array-like, shape = [n_samples]
        y training+test data

    model: estimator object.
        This is assumed to implement the scikit-learn estimator interface.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored.

    niter : int
        number of DCV iteration

    Returns
    -------
    None
    """
    # parameters
    ns_in = 3 # n_splits for inner loop
    ns_ou = 3 # n_splits for outer loop
    scores = np.zeros((niter,5))
    for iiter in range(niter):
        ypreds = np.array([]) # list of predicted y in outer loop
        ytests = np.array([]) # list of y_test in outer loop
        kf_ou = KFold(n_splits=ns_ou, shuffle=True)
    
        # [start] outer loop for test of the generalization error
        for train_index, test_index in kf_ou.split(X):
            X_train, X_test = X[train_index], X[test_index] # inner loop CV
            y_train, y_test = y[train_index], y[test_index] # outer loop 
        
            # [start] inner loop CV for hyper parameter optimization
            kf_in = KFold(n_splits=ns_in, shuffle=True)
            gscv = GridSearchCV(model, param_grid, cv=kf_in)
            gscv.fit(X_train, y_train)
            # [end] inner loop CV for hyper parameter optimization
            
            # test of the generalization error
            ypred = gscv.predict(X_test)
            ypreds = np.append(ypreds, ypred)
            ytests = np.append(ytests, y_test)
        
        # [end] outer loop for test of the generalization error
        tn, fp, fn, tp = confusion_matrix(ytests, ypreds).ravel()
        acc = accuracy_score(ytests, ypreds)
        scores[iiter,:] = np.array([tp,fp,fn,tn,acc])

    means, stds = np.mean(scores, axis=0),np.std(scores, axis=0)
    print()
    print('Double Cross Validation')
    print('In {:} iterations, average +/- standard deviation'.format(niter))
    print('TP   DCV: {:.3f} (+/-{:.3f})'.format(means[0], stds[0]))
    print('FP   DCV: {:.3f} (+/-{:.3f})'.format(means[1], stds[1]))
    print('FN   DCV: {:.3f} (+/-{:.3f})'.format(means[2], stds[2]))
    print('TN   DCV: {:.3f} (+/-{:.3f})'.format(means[3], stds[3]))
    print('Acc. DCV: {:.3f} (+/-{:.3f})'.format(means[4], stds[4]))




def optimize_gamma(X, gammas):
    """
    Optimize gamma by maximizing variance in Gram matrix
    
    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        X training+test data

    gammas : list
        list of gammas

    Returns
    -------
    real
        optimized gamma
    """
    var_matrix = list()
    for gamma in gammas:
        gram_matrix = np.exp(-gamma*((X[:, np.newaxis] - X)**2).sum(axis=2))
        var_matrix.append(gram_matrix.var(ddof=1))
    return gammas[ np.where( var_matrix == np.max(var_matrix) )[0][0] ]




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
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X_train)
    dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
    y_appd = 2 * (dist < ad_thr) -1

    # ver.2
    if(False):    
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(X_train, X_train)
        dist_matrix.sort()
        dist_list = np.mean(dist_matrix[:, 1:n_neighbors+1], axis=1)
        dist_list.sort()
        ad_thr = dist_list[round(X_train.shape[0] * r_ad) - 1]
        dist_matrix = cdist(X_test, X_train)
        dist_matrix.sort()
        dist = np.mean(dist_matrix[:, 0:n_neighbors], axis=1)
        y_appd2 = 2 * (dist < ad_thr) -1
        print(np.allclose(y_appd,y_appd2))
    return y_appd




def ad_ocsvm(X_train, X_test):
    """
    Determination of Applicability Domains (One-Class Support Vector Machine)

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
    range_g = 2**np.arange( -20, 11, dtype=float)
    optgamma = optimize_gamma(X_train, range_g) 
    clf = OneClassSVM(nu=0.003, gamma=optgamma)
    clf.fit(X_train)
    y_appd = clf.predict(X_test) # outliers = -1
    return y_appd


def y_randamization_rgr(X,y,model,param_grid,niter):
    # parameters
    scores = np.zeros((niter,3))
    for iiter in range(niter):
        y_rand = np.random.permutation(y)
        gscv = GridSearchCV(model, param_grid, cv=KFold(n_splits=3, shuffle=True))
        gscv.fit(X, y_rand)
        y_pred = gscv.predict(X)
        rmse  = np.sqrt(mean_squared_error (y_rand, y_pred))
        mae   =         mean_absolute_error(y_rand, y_pred)
        r2    =         r2_score           (y_rand, y_pred)
        scores[iiter,:] = np.array([rmse,mae,r2])
    means, stds = np.mean(scores, axis=0),np.std(scores, axis=0)
    print()
    print("y-randomization")
    print('In {:} iterations, average +/- standard deviation'.format(niter))
#    print('RMSE: {:6.3f} (+/-{:.3f})'.format(means[0], stds[0]))
#    print('MAE : {:6.3f} (+/-{:.3f})'.format(means[1], stds[1]))
#    print('R^2 : {:6.3f} (+/-{:.3f})'.format(means[2], stds[2]))
    print('rnd:RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f} (ave)'\
          .format(means[0], means[1], means[2]))
    print('rnd:RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f} (std)'\
          .format(stds[0], stds[1], stds[2]))
    return

if __name__ == '__main__':
    print('Hello world')
    