# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:17:06 2018

@author: Akitaka
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics         import mean_absolute_error
from sklearn.metrics         import mean_squared_error
from sklearn.metrics         import r2_score

def print_gscv_score(gscv): #{{{
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


def print_score(y_test,y_pred):
    rmse  = np.sqrt(mean_squared_error (y_test,y_pred))
    mae   =         mean_absolute_error(y_test,y_pred)
    if(mae > 0):
        rmae = np.sqrt(mean_squared_error (y_test,y_pred))/mae
    else:
        rmae = 0.0
    r2    =         r2_score           (y_test,y_pred)
    print('RMSE, MAE, RMSE/MAE, R^2 = {:.3f}, {:.3f}, {:.3f}, {:.3f}'\
    .format(rmse, mae, rmae, r2))
# ref: RMSE/MAE
# https://funatsu-lab.github.io/open-course-ware/basic-theory/accuracy-index/


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
    
    # Check: error follows a normal distribution?
    # ref:
    # http://univprof.com/archives/16-07-20-4857140.html
    plt.subplot(1,2,2)
    error = np.array(y_pred-y_obs)
    plt.hist(error)
    plt.title("Gaussian?")
    plt.xlabel('prediction error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    return fig
