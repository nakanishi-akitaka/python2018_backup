# -*- coding: utf-8 -*-
"""
ref:
https://datachemeng.com/variabletransformationad/

@author: Hiromasa Kaneko
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from sklearn import model_selection, svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV

# Settings
regression_flag = 2 # 1:PLS, 2:SVR
Xvariable_transform_flag = 1 # 0:no transform, 1: correct transform

maxPLScomponentnumber = 2
svr_cs = 2 ** np.arange(-5, 10, dtype=float)  # Candidates of C
svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Candidates of epsilon
svr_gammas = 2 ** np.arange(-20, 10, dtype=float)  # Candidates of gamma
fold_number = 5  # "fold_number"-fold cross-validation
number_of_training_samples = 500
number_of_test_samples = 500

# Generate samples for demonstration
np.random.seed(seed=100)
Xtrain_raw = np.random.rand(number_of_training_samples, 2) * 10 - 5
Xtest_raw = np.random.rand(number_of_test_samples, 2) * 14 - 7
Xtest_raw_inad = np.random.rand(number_of_test_samples, 2) * 10 - 5

ytrain = 3 * np.exp(Xtrain_raw[:, 0:1]) + 2 * Xtrain_raw[:, 1:2] ** 3
y_noise_std = ytrain.std(ddof=1) * 0.05
ytrain = ytrain + y_noise_std * np.random.randn(number_of_training_samples, 1)
ytest = 3 * np.exp(Xtest_raw[:, 0:1]) + 2 * Xtest_raw[:, 1:2] ** 3 + y_noise_std * np.random.randn(number_of_test_samples, 1)
ytest_inad = 3 * np.exp(Xtest_raw_inad[:, 0:1]) + 2 * Xtest_raw_inad[:, 1:2] ** 3 + y_noise_std * np.random.randn(number_of_test_samples, 1)
ytrain = np.ndarray.flatten(ytrain)
ytest = np.ndarray.flatten(ytest)
ytest_inad= np.ndarray.flatten(ytest_inad)

Xtrain = Xtrain_raw
Xtest = Xtest_raw
Xtest_inad = Xtest_raw_inad
if Xvariable_transform_flag == 1:
    Xtrain[:, 0] = np.exp(Xtrain_raw[:, 0])
    Xtrain[:, 1] = Xtrain_raw[:, 1] ** 3
    Xtest[:, 0] = np.exp(Xtest_raw[:, 0])
    Xtest[:, 1] = Xtest_raw[:, 1] ** 3
    Xtest_inad[:, 0] = np.exp(Xtest_raw_inad[:, 0])
    Xtest_inad[:, 1] = Xtest_raw_inad[:, 1] ** 3
    
# Standarize X and y
autoscaled_Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
autoscaled_ytrain = (ytrain - ytrain.mean()) / ytrain.std(ddof=1)
autoscaled_Xtest = (Xtest - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
autoscaled_Xtest_inad = (Xtest_inad - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)

if regression_flag == 1:
    PLScomponents = np.arange( 1, min(np.linalg.matrix_rank(autoscaled_Xtrain)+1,maxPLScomponentnumber+1), 1)
    r2all = list()
    r2cvall = list()
    for PLScomponent in PLScomponents:
        plsmodelincv = PLSRegression(n_components=PLScomponent)
        plsmodelincv.fit(autoscaled_Xtrain, autoscaled_ytrain)
        calculatedyincv = np.ndarray.flatten( plsmodelincv.predict(autoscaled_Xtrain) )
        estimatedyincv = np.ndarray.flatten( model_selection.cross_val_predict(plsmodelincv, autoscaled_Xtrain, autoscaled_ytrain, cv=fold_number) )
        calculatedyincv = calculatedyincv*ytrain.std(ddof=1) + ytrain.mean()
        estimatedyincv = estimatedyincv*ytrain.std(ddof=1) + ytrain.mean()
            
        r2all.append( float( 1 - sum( (ytrain-calculatedyincv )**2 ) / sum((ytrain-ytrain.mean())**2) ))
        r2cvall.append( float( 1 - sum( (ytrain-estimatedyincv )**2 ) / sum((ytrain-ytrain.mean())**2) ))
    plt.plot( PLScomponents, r2all, 'bo-')
    plt.plot( PLScomponents, r2cvall, 'ro-')
    plt.ylim(0,1)
    plt.xlabel('Number of PLS components')
    plt.ylabel('r2(blue), r2cv(red)')
    plt.show()
    optimal_PLS_componentnumber = np.where( r2cvall == np.max(r2cvall) )
    optimal_PLS_componentnumber = optimal_PLS_componentnumber[0][0]+1
    regression_model = PLSRegression(n_components=optimal_PLS_componentnumber)
elif regression_flag == 2:
    variance_of_gram_matrix = list()
    for svr_gamma in svr_gammas:
        gram_matrix = np.exp(
            -svr_gamma * ((autoscaled_Xtrain[:, np.newaxis] - autoscaled_Xtrain) ** 2).sum(axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_svr_gamma = svr_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
    
    # Optimize epsilon with cross-validation
    svr_model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                                   cv=fold_number)
    svr_model_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
    optimal_svr_epsilon = svr_model_in_cv.best_params_['epsilon']
    
    # Optimize C with cross-validation
    svr_model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                                   {'C': svr_cs}, cv=fold_number)
    svr_model_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
    optimal_svr_c = svr_model_in_cv.best_params_['C']
    
    # Optimize gamma with cross-validation (optional)
    svr_model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                                   {'gamma': svr_gammas}, cv=fold_number)
    svr_model_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
    optimal_svr_gamma = svr_model_in_cv.best_params_['gamma']
    
    # Check optimized hyperparameters
    print("C: {0}, Epsion: {1}, Gamma: {2}".format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))
    
    # Construct SVR model
    regression_model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)

regression_model.fit(autoscaled_Xtrain, autoscaled_ytrain)

# Calculate y of training dataset
calculated_ytrain = np.ndarray.flatten(regression_model.predict(autoscaled_Xtrain))
calculated_ytrain = calculated_ytrain * ytrain.std(ddof=1) + ytrain.mean()
# r2, RMSE, MAE
print("r2: {0}".format(float(1 - sum((ytrain - calculated_ytrain) ** 2) / sum((ytrain - ytrain.mean()) ** 2))))
print("RMSE: {0}".format(float((sum((ytrain - calculated_ytrain) ** 2) / len(ytrain)) ** 0.5)))
print("MAE: {0}".format(float(sum(abs(ytrain - calculated_ytrain)) / len(ytrain))))
# yy-plot
plt.rcParams["font.size"] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytrain, calculated_ytrain)
YMax = np.max(np.array([np.array(ytrain), calculated_ytrain]))
YMin = np.min(np.array([np.array(ytrain), calculated_ytrain]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("Actual Y")
plt.ylabel("Calculated Y")
plt.show()

# Estimate y in cross-validation
estimated_y_in_cv = np.ndarray.flatten(
    model_selection.cross_val_predict(regression_model, autoscaled_Xtrain, autoscaled_ytrain, cv=fold_number))
estimated_y_in_cv = estimated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()
# r2cv, RMSEcv, MAEcv
print("r2cv: {0}".format(float(1 - sum((ytrain - estimated_y_in_cv) ** 2) / sum((ytrain - ytrain.mean()) ** 2))))
print("RMSEcv: {0}".format(float((sum((ytrain - estimated_y_in_cv) ** 2) / len(ytrain)) ** 0.5)))
print("MAEcv: {0}".format(float(sum(abs(ytrain - estimated_y_in_cv)) / len(ytrain))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytrain, estimated_y_in_cv)
YMax = np.max(np.array([np.array(ytrain), estimated_y_in_cv]))
YMin = np.min(np.array([np.array(ytrain), estimated_y_in_cv]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y in CV")
plt.show()

# Estimate y of test dataset
predicted_ytest = np.ndarray.flatten(regression_model.predict(autoscaled_Xtest))
predicted_ytest = predicted_ytest * ytrain.std(ddof=1) + ytrain.mean()
# r2p, RMSEp, MAEp
print("r2p: {0}".format(float(1 - sum((ytest - predicted_ytest) ** 2) / sum((ytest - ytest.mean()) ** 2))))
print("RMSEp: {0}".format(float((sum((ytest - predicted_ytest) ** 2) / len(ytest)) ** 0.5)))
print("MAEp: {0}".format(float(sum(abs(ytest - predicted_ytest)) / len(ytest))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytest, predicted_ytest)
YMax = np.max(np.array([np.array(ytest), predicted_ytest]))
YMin = np.min(np.array([np.array(ytest), predicted_ytest]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y")
plt.show()

# Estimate y of test dataset (inside AD)
predicted_ytest_inad = np.ndarray.flatten(regression_model.predict(autoscaled_Xtest_inad))
predicted_ytest_inad = predicted_ytest_inad * ytrain.std(ddof=1) + ytrain.mean()
# r2p, RMSEp, MAEp
print("r2p: {0}".format(float(1 - sum((ytest_inad - predicted_ytest_inad) ** 2) / sum((ytest_inad - ytest_inad.mean()) ** 2))))
print("RMSEp: {0}".format(float((sum((ytest_inad - predicted_ytest_inad) ** 2) / len(ytest_inad)) ** 0.5)))
print("MAEp: {0}".format(float(sum(abs(ytest_inad - predicted_ytest_inad)) / len(ytest_inad))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytest_inad, predicted_ytest_inad)
YMax = np.max(np.array([np.array(ytest_inad), predicted_ytest_inad]))
YMin = np.min(np.array([np.array(ytest_inad), predicted_ytest_inad]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y")
plt.show()