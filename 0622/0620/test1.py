#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV

regression_method_flag = 8  # 1:OLS, 2:PLS(constant component number), 3:PLS, 4:Ridge regression
# 5:LASSO, 6:Elastic net, 7:Linear SVR, 8:Nonlinear SVR, 9:Random forest
number_of_training_data         = 878   # if this is the number of all samples, there are no test samples.
do_autoscaling                  = True  # True or False
threshold_of_rate_of_same_value = 0.79
fold_number                     = 2
pls_component_number            = 2
max_pls_component_number        = 50
ridge_lambdas                   = 2 ** np.arange(-5, 10, dtype=float)       # L2 weight in ridge regression
lasso_lambdas                   = np.arange(0.01, 0.71, 0.01, dtype=float)  # L1 weight in LASSO
elastic_net_lambdas             = np.arange(0.01, 0.71, 0.01, dtype=float)  # Lambda in elastic net
elastic_net_alphas              = np.arange(0.01, 1.00, 0.01, dtype=float)  # Alpha in elastic net
linear_svr_cs                   = 2 ** np.arange(-5, 5, dtype=float)        # C for linear svr
linear_svr_epsilons             = 2 ** np.arange(-10, 0, dtype=float)       # Epsilon for linear svr
nonlinear_svr_cs                = 2 ** np.arange(-5, 10, dtype=float)       # C for nonlinear svr
nonlinear_svr_epsilons          = 2 ** np.arange(-10, 0, dtype=float)       # Epsilon for nonlinear svr
nonlinear_svr_gammas            = 2 ** np.arange(-20, 10, dtype=float)      # Gamma for nonlinear svr
random_forest_number_of_trees   = 300                                       # Number of decision trees for random forest
random_forest_x_variables_rates = np.arange(1, 10, dtype=float) / 10        # Ratio of the number of X-variables for random forest

# load data set
raw_data_with_y = pd.read_csv('dataset/logSdataset1290.csv', encoding='SHIFT-JIS', index_col=0)

raw_data_with_y = raw_data_with_y.loc[:, raw_data_with_y.mean().index]  # pick up mean-calculatable parameters
# raw_data_with_y = raw_data_with_y.loc[raw_data_with_y.mean(axis=1).index,:] # pick up mean-calculatable samples

raw_data_with_y = raw_data_with_y.replace(np.inf, np.nan).fillna(np.nan)  # inf -> NaN
raw_data_with_y = raw_data_with_y.dropna(axis=1)  # delete parameters including NaN
# raw_data_with_y = raw_data_with_y.dropna() # delete samples including NaN

# pick up parameters for logSdataset1290.csv
raw_data_with_y = raw_data_with_y.drop(['Ipc', 'Kappa3'], axis=1)  # Ipc:1139 is hazue, Kappa3:889 is hazure

# delete duplicates
# raw_data_with_y = raw_data_with_y.loc[~raw_data_with_y.index.duplicated(keep='first'),:] # delete dupulicated sample (first)
# raw_data_with_y = raw_data_with_y.loc[~raw_data_with_y.index.duplicated(keep='last'),:]  # delete dupulicated sample (last)
raw_data_with_y = raw_data_with_y.loc[~raw_data_with_y.index.duplicated(keep=False), :]    # delete dupulicated samples (all)

ytrain = raw_data_with_y.iloc[:number_of_training_data, 0]
raw_Xtrain = raw_data_with_y.iloc[:number_of_training_data, 1:]
ytest = raw_data_with_y.iloc[number_of_training_data:, 0]
raw_Xtest = raw_data_with_y.iloc[number_of_training_data:, 1:]

# y = raw_data_with_y[raw_data_with_y.columns[0]]
# rawX = raw_data_with_y[raw_data_with_y.columns[1:]]
# rawX_tmp = rawX.copy()

# delete descriptors with high rate of the same values
rate_of_same_value = list()
num = 0
for X_variable_name in raw_Xtrain.columns:
    num += 1
    print('{0} / {1}'.format(num, raw_Xtrain.shape[1]))
    same_value_number = raw_Xtrain[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / raw_Xtrain.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)

"""
# delete descriptors with zero variance
deleting_variable_numbers = np.where( raw_Xtrain.var() == 0 )
"""

if len(deleting_variable_numbers[0]) == 0:
    Xtrain = raw_Xtrain.copy()
    Xtest = raw_Xtest.copy()
else:
    Xtrain = raw_Xtrain.drop(raw_Xtrain.columns[deleting_variable_numbers], axis=1)
    Xtest = raw_Xtest.drop(raw_Xtest.columns[deleting_variable_numbers], axis=1)
    print('Variable numbers zero variance: {0}'.format(deleting_variable_numbers[0] + 1))

print('# of X-variables: {0}'.format(Xtrain.shape[1]))

# autoscaling
if do_autoscaling:
    autoscaled_Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
    autoscaled_ytrain = (ytrain - ytrain.mean()) / ytrain.std(ddof=1)
    autoscaled_Xtest = (Xtest - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
else:
    autoscaled_Xtrain = Xtrain.copy()
    autoscaled_ytrain = ytrain.copy()
    autoscaled_Xtest = Xtest.copy()

if regression_method_flag == 1:  # Ordinary Least Squares
    regression_model = LinearRegression()
elif regression_method_flag == 2:  # Partial Least Squares with constant component
    regression_model = PLSRegression(n_components=pls_component_number)
elif regression_method_flag == 3:  # Partial Least Squares
    pls_components = np.arange(1, min(np.linalg.matrix_rank(autoscaled_Xtrain) + 1, max_pls_component_number + 1), 1)
    r2all = list()
    r2cvall = list()
    for pls_component in pls_components:
        pls_model_in_cv = PLSRegression(n_components=pls_component)
        pls_model_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
        calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(autoscaled_Xtrain))
        estimated_y_in_cv = np.ndarray.flatten(
            model_selection.cross_val_predict(pls_model_in_cv, autoscaled_Xtrain, autoscaled_ytrain, cv=fold_number))
        if do_autoscaling:
            calculated_y_in_cv = calculated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()
            estimated_y_in_cv = estimated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()

        """
        plt.figure(figsize=figure.figaspect(1))
        plt.scatter( y, estimated_y_in_cv)
        plt.xlabel("Actual Y")
        plt.ylabel("Calculated Y")
        plt.show()
        """
        r2all.append(float(1 - sum((ytrain - calculated_y_in_cv) ** 2) / sum((ytrain - ytrain.mean()) ** 2)))
        r2cvall.append(float(1 - sum((ytrain - estimated_y_in_cv) ** 2) / sum((ytrain - ytrain.mean()) ** 2)))
    plt.plot(pls_components, r2all, 'bo-')
    plt.plot(pls_components, r2cvall, 'ro-')
    plt.ylim(0, 1)
    plt.xlabel('Number of PLS components')
    plt.ylabel('r2(blue), r2cv(red)')
    plt.show()
    optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))
    optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
    regression_model = PLSRegression(n_components=optimal_pls_component_number)
elif regression_method_flag == 4:  # ridge regression
    r2cvall = list()
    for ridge_lambda in ridge_lambdas:
        rr_model_in_cv = Ridge(alpha=ridge_lambda)
        estimated_y_in_cv = model_selection.cross_val_predict(rr_model_in_cv, autoscaled_Xtrain, autoscaled_ytrain,
                                                              cv=fold_number)
        if do_autoscaling:
            estimated_y_in_cv = estimated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()
        r2cvall.append(float(1 - sum((ytrain - estimated_y_in_cv) ** 2) / sum((ytrain - ytrain.mean()) ** 2)))
    plt.figure()
    plt.plot(ridge_lambdas, r2cvall, 'k', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Weight for ridge regression')
    plt.ylabel('r2cv for ridge regression')
    plt.show()
    optimal_ridge_lambda = ridge_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
    regression_model = Ridge(alpha=optimal_ridge_lambda)
elif regression_method_flag == 5:  # LASSO
    r2cvall = list()
    for lasso_lambda in lasso_lambdas:
        lasso_model_in_cv = Lasso(alpha=lasso_lambda)
        estimated_y_in_cv = model_selection.cross_val_predict(lasso_model_in_cv, autoscaled_Xtrain, autoscaled_ytrain,
                                                              cv=fold_number)
        if do_autoscaling:
            estimated_y_in_cv = estimated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()
        r2cvall.append(float(1 - sum((ytrain - estimated_y_in_cv) ** 2) / sum((ytrain - ytrain.mean()) ** 2)))
    plt.figure()
    plt.plot(lasso_lambdas, r2cvall, 'k', linewidth=2)
    plt.xlabel('Weight for LASSO')
    plt.ylabel('r2cv for LASSO')
    plt.show()
    optimal_lasso_lambda = lasso_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
    regression_model = Lasso(alpha=optimal_lasso_lambda)
elif regression_method_flag == 6:  # Elastic net
    elastic_net_in_cv = ElasticNetCV(cv=fold_number, l1_ratio=elastic_net_lambdas, alphas=elastic_net_alphas)
    elastic_net_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
    optimal_elastic_net_alpha = elastic_net_in_cv.alpha_
    optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_
    regression_model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)
elif regression_method_flag == 7:  # Linear SVR
    linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_cs, 'epsilon': linear_svr_epsilons},
                                    cv=fold_number)
    linear_svr_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
    optimal_linear_svr_c = linear_svr_in_cv.best_params_['C']
    optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon']
    regression_model = svm.SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon)
elif regression_method_flag == 8:  # Nonlinear SVR
    variance_of_gram_matrix = list()
    numpy_autoscaled_Xtrain = np.array(autoscaled_Xtrain)
    for nonlinear_svr_gamma in nonlinear_svr_gammas:
        gram_matrix = np.exp(
            -nonlinear_svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(
                axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_nonlinear_gamma = nonlinear_svr_gammas[
        np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
    nonlinear_svr_in_cv = GridSearchCV(svm.SVR(kernel='rbf', gamma=optimal_nonlinear_gamma),
                                       {'C': nonlinear_svr_cs, 'epsilon': nonlinear_svr_epsilons}, cv=fold_number)
    nonlinear_svr_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
    optimal_nonlinear_c = nonlinear_svr_in_cv.best_params_['C']
    optimal_nonlinear_epsilon = nonlinear_svr_in_cv.best_params_['epsilon']
    regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                               gamma=optimal_nonlinear_gamma)
elif regression_method_flag == 9:  # Random forest
    rmse_oob_all = list()
    for random_forest_x_variables_rate in random_forest_x_variables_rates:
        RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
            max(math.ceil(Xtrain.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
        RandomForestResult.fit(autoscaled_Xtrain, autoscaled_ytrain)
        estimated_y_in_cv = RandomForestResult.oob_prediction_
        if do_autoscaling:
            estimated_y_in_cv = estimated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()
        rmse_oob_all.append((sum((ytrain - estimated_y_in_cv) ** 2) / len(ytrain)) ** 0.5)
    plt.figure()
    plt.plot(random_forest_x_variables_rates, rmse_oob_all, 'k', linewidth=2)
    plt.xlabel('Ratio of the number of X-variables')
    plt.ylabel('RMSE of OOB')
    plt.show()
    optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
        np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]]
    regression_model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
        max(math.ceil(Xtrain.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True)

regression_model.fit(autoscaled_Xtrain, autoscaled_ytrain)

# calculate y
calculated_ytrain = np.ndarray.flatten(regression_model.predict(autoscaled_Xtrain))
if do_autoscaling:
    calculated_ytrain = calculated_ytrain * ytrain.std(ddof=1) + ytrain.mean()
# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((ytrain - calculated_ytrain) ** 2) / sum((ytrain - ytrain.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((ytrain - calculated_ytrain) ** 2) / len(ytrain)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(ytrain - calculated_ytrain)) / len(ytrain))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytrain, calculated_ytrain)
YMax = np.max(np.array([np.array(ytrain), calculated_ytrain]))
YMin = np.min(np.array([np.array(ytrain), calculated_ytrain]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()

# estimated_y in cross-validation
estimated_y_in_cv = np.ndarray.flatten(
    model_selection.cross_val_predict(regression_model, autoscaled_Xtrain, autoscaled_ytrain, cv=fold_number))
if do_autoscaling:
    estimated_y_in_cv = estimated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()
# r2cv, RMSEcv, MAEcv
print('r2cv: {0}'.format(float(1 - sum((ytrain - estimated_y_in_cv) ** 2) / sum((ytrain - ytrain.mean()) ** 2))))
print('RMSEcv: {0}'.format(float((sum((ytrain - estimated_y_in_cv) ** 2) / len(ytrain)) ** 0.5)))
print('MAEcv: {0}'.format(float(sum(abs(ytrain - estimated_y_in_cv)) / len(ytrain))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytrain, estimated_y_in_cv)
YMax = np.max(np.array([np.array(ytrain), estimated_y_in_cv]))
YMin = np.min(np.array([np.array(ytrain), estimated_y_in_cv]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y in CV')
plt.show()

# standard regression coefficients
# standard_regression_coefficients = regression_model.coef_
# standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients)
# standard_regression_coefficients.index = Xtrain.columns
# standard_regression_coefficients.columns = ['standard regression coefficient']
# standard_regression_coefficients.to_csv( 'standard_regression_coefficients.csv' )

# prediction
if raw_Xtest.shape[0]:
    predicted_ytest = np.ndarray.flatten(regression_model.predict(autoscaled_Xtest))
    if do_autoscaling:
        predicted_ytest = predicted_ytest * ytrain.std(ddof=1) + ytrain.mean()
    # r2p, RMSEp, MAEp
    print('r2p: {0}'.format(float(1 - sum((ytest - predicted_ytest) ** 2) / sum((ytest - ytest.mean()) ** 2))))
    print('RMSEp: {0}'.format(float((sum((ytest - predicted_ytest) ** 2) / len(ytest)) ** 0.5)))
    print('MAEp: {0}'.format(float(sum(abs(ytest - predicted_ytest)) / len(ytest))))
    # yy-plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(ytest, predicted_ytest)
    YMax = np.max(np.array([np.array(ytest), predicted_ytest]))
    YMin = np.min(np.array([np.array(ytest), predicted_ytest]))
    plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
             [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
    plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
    plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.show()
