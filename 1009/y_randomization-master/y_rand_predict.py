# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
import numpy as np
from sklearn import model_selection

def y_rand_predict(model, x_train, y_train, cv, seed):
    """
    Locally-Weighted Partial Least Squares (LWPLS)
    
    Predict y-values of test samples using LWPLS

    Parameters
    ----------
    model: model in sklearn before fitting
    x_train: numpy.array or pandas.DataFrame
        autoscaled m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y_train: numpy.array or pandas.DataFrame
        autoscaled m x 1 vector of a Y-variable of training data
    cv: int
        number of fold, which is the same as 'cv' in sklearn
    seed: int
        random seed, if seed = -999, random seed is not set

    Returns
    -------
    y_train_rand : numpy.array
        k x 1 vector of randomized y-values of training data
    estimated_y_train_rand : numpy.array
        k x 1 vector of estimated y-values of randomized training data
    estimated_y_train_in_cv_rand : numpy.array
        k x 1 vector estimated of y-values of randomized training data in cross-validation
    """

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    if seed != -999:
        np.random.seed(seed)
    y_train_rand = np.random.permutation(y_train)
    autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_y_train_rand = (y_train_rand - y_train_rand.mean()) / y_train_rand.std(ddof=1)
    
    model.fit(autoscaled_x_train, autoscaled_y_train_rand)
    estimated_y_train_rand = np.ndarray.flatten(model.predict(autoscaled_x_train))
    estimated_y_train_rand = estimated_y_train_rand * y_train_rand.std(ddof=1) + y_train_rand.mean()
    estimated_y_train_in_cv_rand = np.ndarray.flatten(
        model_selection.cross_val_predict(model, autoscaled_x_train, autoscaled_y_train_rand,
                                          cv=cv))
    estimated_y_train_in_cv_rand = estimated_y_train_in_cv_rand * y_train_rand.std(ddof=1) + y_train_rand.mean()
    
    return y_train_rand, estimated_y_train_rand, estimated_y_train_in_cv_rand
