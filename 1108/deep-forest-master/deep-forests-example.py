# -*- coding: utf-8 -*-
"""
https://github.com/leopiney/deep-forest/
Created on Thu Nov  8 13:47:11 2018

@author: Akitaka
"""

#
# Inpired by https://arxiv.org/abs/1702.08835 and https://github.com/STO-OTZ/my_gcForest/
#
import numpy as np
import random
import uuid

from sklearn.datasets import fetch_mldata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from deep_forest import MGCForest

#%%
#from sklearn.datasets.base import get_data_home 
#print (get_data_home()) # これで表示されるパスに mnist-original.mat を置く
#%%
# The MNIST dataset
# mnist = fetch_mldata('MNIST original', data_home='.')
mnist = fetch_mldata('MNIST original')
mnist.data.shape

print('Data: {}, target: {}'.format(mnist.data.shape, mnist.target.shape))

X_train, X_test, y_train, y_test = train_test_split(
    mnist.data,
    mnist.target,
    test_size=0.2,
    random_state=42,
)

X_train = X_train.reshape((len(X_train), 28, 28))
X_test = X_test.reshape((len(X_test), 28, 28))

#
# Limit the size of the dataset
#
#X_train = X_train[:2000]
#y_train = y_train[:2000]
#X_test = X_test[:2000]
#y_test = y_test[:2000]
X_train = X_train[:200]
y_train = y_train[:200]
X_test = X_test[:200]
y_test = y_test[:200]

print('X_train:', X_train.shape, X_train.dtype)
print('y_train:', y_train.shape, y_train.dtype)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

#%%
# Using the MGCForest
# Creates a simple MGCForest with
# 2 random forests for the Multi-Grained-Scanning process
# and 2 other random forests for the Cascade process.
mgc_forest = MGCForest(
    estimators_config={
        'mgs': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 30,
                'min_samples_split': 21,
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 30,
                'min_samples_split': 21,
            }
        }],
        'cascade': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 1,
            }
        }, {
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 'sqrt',
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 1,
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 'sqrt',
            }
        }]
    },
    stride_ratios=[1.0 / 4, 1.0 / 9, 1.0 / 16],
)

mgc_forest.fit(X_train, y_train)

y_pred = mgc_forest.predict(X_test)

print('Prediction shape:', y_pred.shape)
print(
    'Accuracy:', accuracy_score(y_test, y_pred),
    'F1 score:', f1_score(y_test, y_pred, average='weighted')
)
