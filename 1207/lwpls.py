# -*- coding: utf-8 -*-
"""
https://github.com/hkaneko1985/lwpls/blob/master/Python/lwpls.py

Created on Fri Dec  7 14:50:21 2018

@author: Akitaka
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin

class LWPLS(BaseEstimator, RegressorMixin):
    """
    Locally-Weighted Partial Least Squares (LWPLS)
    
    Predict y-values of test samples using LWPLS
    x_test: numpy.array or pandas.DataFrame
        k x n matrix of X-variables of test data, which is autoscaled with training data,
        and k is the number of test samples
    max_component_number: int
        number of maximum components
    lambda_in_similarity: float
        parameter in similarity matrix
    """


    def __init__(self, n_components=2, l_similarity=2**-2):
        self.train_data = None
        self.target_data = None
        self.n_components = n_components
        self.l_similarity = l_similarity

    def fit(self, X, y):
        """
        Fit model to data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, n_targets]
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables
.        """
        self.train_data = np.array(X)
        self.target_data = np.array(y)

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array of shape = [n_samples]
            The predicted values.
        """
        X_train = self.train_data
        y_train = np.reshape(self.target_data, (len(self.target_data), 1))
        X_test = np.array(X)
        n_components = self.n_components
        l_similarity = self.l_similarity
        
        y_pred = np.zeros((X_test.shape[0], n_components))
        distance_matrix = cdist(X_train, X_test, 'euclidean')
        for test_sample_number in range(X_test.shape[0]):
            query_x_test = X_test[test_sample_number, :]
            query_x_test = np.reshape(query_x_test, (1, len(query_x_test)))
            distance = distance_matrix[:, test_sample_number]
            similarity = np.diag(np.exp(-distance / distance.std(ddof=1) / l_similarity))
            #        similarity_matrix = np.diag(similarity)
    
            y_w = y_train.T.dot(np.diag(similarity)) / similarity.sum()
            x_w = np.reshape(X_train.T.dot(np.diag(similarity)) / similarity.sum(), (1, X_train.shape[1]))
            centered_y = y_train - y_w
            centered_x = X_train - np.ones((X_train.shape[0], 1)).dot(x_w)
            centered_query_x_test = query_x_test - x_w
            y_pred[test_sample_number, :] += y_w
            for component_number in range(n_components):
                w_a = np.reshape(centered_x.T.dot(similarity).dot(centered_y) / np.linalg.norm(
                    centered_x.T.dot(similarity).dot(centered_y)), (X_train.shape[1], 1))
                t_a = np.reshape(centered_x.dot(w_a), (X_train.shape[0], 1))
                p_a = np.reshape(centered_x.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a),
                                 (X_train.shape[1], 1))
                q_a = centered_y.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a)
                t_q_a = centered_query_x_test.dot(w_a)
                y_pred[test_sample_number, component_number:] = y_pred[test_sample_number,
                      component_number:] + t_q_a * q_a
                if component_number != n_components:
                    centered_x = centered_x - t_a.dot(p_a.T)
                    centered_y = centered_y - t_a * q_a
                    centered_query_x_test = centered_query_x_test - t_q_a.dot(p_a.T)

        return y_pred[:, n_components-1]
    