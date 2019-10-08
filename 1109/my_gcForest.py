# -*- coding: utf-8 -*-
"""
https://github.com/STO-OTZ/my_gcForest/
Created on Thu Nov  8 14:26:05 2018

@author: Akitaka
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict as cvp
import random
from functools import reduce

class MultiGrainedScaner():
    def __init__(self, base_estimator, params_list, sliding_ratio = 0.25, k_fold = 3):
        if k_fold > 1: #use cv
            self.params_list = params_list
        else:#use oob
            self.params_list = [params.update({'oob_score':True}) or params for params in params_list]
        self.sliding_ratio = sliding_ratio
        self.k_fold = k_fold
        self.base_estimator = base_estimator
        klass = self.base_estimator.__class__
        self.estimators = [klass(**params) for params in self.params_list]

    #generate scaned samples, X is not None, X[0] is no more than 3d
    def _sample_slicer(self,X,y):
        data_shape = X[0].shape
        window_shape = [max(int(data_size * self.sliding_ratio),1) for data_size in data_shape]
        scan_round_axis = [data_shape[i]-window_shape[i]+1 for i in range(len(data_shape))]
        scan_round_total = reduce(lambda acc,x: acc*x,scan_round_axis)
        if len(data_shape) == 1:
            newX = np.array([x[beg:beg+window_shape[0]]
                                for x in X
                                    for beg in range(scan_round_axis[0])])
        elif len(data_shape) == 2:
            newX = np.array([x[beg0:beg0+window_shape[0],beg1:beg1+window_shape[1]].ravel()
                                for x in X
                                    for beg0 in range(scan_round_axis[0])
                                        for beg1 in range(scan_round_axis[1])])
        elif len(data_shape) == 3:
            newX = np.array([x[beg0:beg0+window_shape[0],beg1:beg1+window_shape[1],beg2:beg2+window_shape[2]].ravel()
                                for x in X
                                    for beg0 in range(scan_round_axis[0])
                                        for beg1 in range(scan_round_axis[1])
                                            for beg2 in range(scan_round_axis[2])])
        newy = y.repeat(scan_round_total)
        return newX,newy,scan_round_total

    #generate new sample vectors
    def scan_fit(self,X,y):
        self.n_classes = len(np.unique(y))
        newX,newy,scan_round_total = self._sample_slicer(X,y)
        sample_vector_list = []
        for estimator in self.estimators:
            estimator.fit(newX, newy)
            if self.k_fold > 1:# use cv
                predict_ = cvp(estimator, newX, newy, cv=self.k_fold)
            else:#use oob
                predict_ = estimator.oob_decision_function_
                #fill default value if meet nan
                inds = np.where(np.isnan(predict_))
                predict_[inds] = 1./self.n_classes
            sample_vector = predict_.reshape((len(X),scan_round_total*self.n_classes))
            sample_vector_list.append(sample_vector)
        return np.hstack(sample_vector_list)

    def scan_predict(self,X):
        newX,newy,scan_round_total = self._sample_slicer(X,np.zeros(len(X)))
        sample_vector_list = []
        for estimator in self.estimators:
            predict_ = estimator.predict(newX)
            sample_vector = predict_.reshape((len(X),scan_round_total*self.n_classes))
            sample_vector_list.append(sample_vector)
        return np.hstack(sample_vector_list)

class CascadeForest():
    def __init__(self, base_estimator, params_list, k_fold = 3, evaluate = lambda pre,y: float(sum(pre==y))/len(y)):
        if k_fold > 1: #use cv
            self.params_list = params_list
        else:#use oob
            self.params_list = [params.update({'oob_score':True}) or params for params in params_list]
        self.k_fold = k_fold
        self.evaluate = evaluate
        self.base_estimator = base_estimator
#         base_class = base_estimator.__class__
#         global prob_class
#         class prob_class(base_class): #to use cross_val_predict, estimator's predict method should be predict_prob
#             def predict(self, X):
#                 return base_class.predict_proba(self, X)
#         self.base_estimator = prob_class()

    def fit(self,X_train,y_train):
        self.n_classes = len(np.unique(y_train))
        self.estimators_levels = []
        klass = self.base_estimator.__class__
        predictions_levels = []
        self.classes = np.unique(y_train)

        #first level
        estimators = [klass(**params) for params in self.params_list]
        self.estimators_levels.append(estimators)
        predictions = []
        for estimator in estimators:
            estimator.fit(X_train, y_train)
            if self.k_fold > 1:# use cv
                predict_ = cvp(estimator, X_train, y_train, cv=self.k_fold)
            else:#use oob
                predict_ = estimator.oob_decision_function_
                #fill default value if meet nan
                inds = np.where(np.isnan(predict_))
                predict_[inds] = 1./self.n_classes
            predictions.append(predict_)
        attr_to_next_level = np.hstack(predictions)
        y_pre = self.classes.take(np.argmax(np.array(predictions).mean(axis=0),axis=1),axis=0)
        self.max_accuracy = self.evaluate(y_pre,y_train)

        #cascade step
        while True:
            print('level {}, CV accuracy: {}'.format(len(self.estimators_levels),self.max_accuracy))
            estimators = [klass(**params) for params in self.params_list]
            self.estimators_levels.append(estimators)
            predictions = []
            X_train_step = np.hstack((attr_to_next_level,X_train))
            for estimator in estimators:
                estimator.fit(X_train_step, y_train)
                if self.k_fold > 1:# use cv
                    predict_ = cvp(estimator, X_train_step, y_train, cv=self.k_fold)
                else:#use oob
                    predict_ = estimator.oob_decision_function_
                    #fill default value if meet nan
                    inds = np.where(np.isnan(predict_))
                    predict_[inds] = 1./self.n_classes
                predictions.append(predict_)
            attr_to_next_level = np.hstack(predictions)
            y_pre = self.classes.take(np.argmax(np.array(predictions).mean(axis=0),axis=1),axis=0)
            accuracy = self.evaluate(y_pre,y_train)
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
            else:
                self.estimators_levels.pop()
                break

    def predict_proba_staged(self,X):
        #init ouput, shape = nlevel * nsample * nclass
        self.proba_staged = np.zeros((len(self.estimators_levels),len(X),self.n_classes))

        #first level
        estimators = self.estimators_levels[0]
        predictions = []
        for estimator in estimators:
            predict_ = estimator.predict(X)
            predictions.append(predict_)
        attr_to_next_level = np.hstack(predictions)
        self.proba_staged[0] = np.array(predictions).mean(axis=0) #不同estimator求平均

        #cascade step
        for i in range(1,len(self.estimators_levels)):
            estimators = self.estimators_levels[i]
            predictions = []
            X_step = np.hstack((attr_to_next_level,X))
            for estimator in estimators:
                predict_ = estimator.predict(X_step)
                predictions.append(predict_)
            attr_to_next_level = np.hstack(predictions)
            self.proba_staged[i] = np.array(predictions).mean(axis=0)

        return self.proba_staged
    
    def predict_proba(self,X):
        return self.predict_proba_staged(X)[-1]
    
    def predict_staged(self,X):
        proba_staged = self.predict_proba_staged(X)
        predictions_staged = np.apply_along_axis(lambda proba: self.classes.take(np.argmax(proba),axis=0),
                                                 2, 
                                                 proba_staged)
        return predictions_staged

    def predict(self,X):
        proba = self.predict_proba(X)
        predictions = self.classes.take(np.argmax(proba,axis=1),axis=0) #平均值最大的index对应的class
        return predictions

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

# Trunk the data
#n_train = 60000
#n_test = 10000
n_train = 600
n_test = 100

# Define training and testing sets
train_idx = np.arange(n_train)
test_idx = np.arange(n_test)+n_train
random.shuffle(train_idx)

X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]

scan_forest_params1 = RandomForestClassifier(n_estimators=30,min_samples_split=21,max_features=1).get_params()
scan_forest_params2 = RandomForestClassifier(n_estimators=30,min_samples_split=21,max_features='sqrt').get_params()

cascade_forest_params1 = RandomForestClassifier(n_estimators=1000,min_samples_split=11,max_features=1).get_params()
cascade_forest_params2 = RandomForestClassifier(n_estimators=1000,min_samples_split=11,max_features='sqrt').get_params()

scan_params_list = [scan_forest_params1,scan_forest_params2]
cascade_params_list = [cascade_forest_params1,cascade_forest_params2]*2

def calc_accuracy(pre,y):
    return float(sum(pre==y))/len(y)
class ProbRandomForestClassifier(RandomForestClassifier):
    def predict(self, X):
        return RandomForestClassifier.predict_proba(self, X)
#%%
# train_size = 1000
train_size = 10

# gcForest 

# Multi-Grained Scan Step
Scaner1 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./4)
Scaner2 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./9)
Scaner3 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./16)

X_train_scan =np.hstack([scaner.scan_fit(X_train[:train_size].reshape((train_size,28,28)), y_train[:train_size])
                             for scaner in [Scaner1,Scaner2,Scaner3][:1]])
X_test_scan = np.hstack([scaner.scan_predict(X_test.reshape((len(X_test),28,28)))
                             for scaner in [Scaner1,Scaner2,Scaner3][:1]])

# Cascade RandomForest Step
CascadeRF = CascadeForest(ProbRandomForestClassifier(),cascade_params_list)
CascadeRF.fit(X_train_scan, y_train[:train_size])
y_pre_staged = CascadeRF.predict_staged(X_test_scan)
test_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,y_test), 1, y_pre_staged)
print('\n'.join('level {}, test accuracy: {}'.format(i+1,test_accuracy_staged[i]) for i in range(len(test_accuracy_staged))))

# CascadeRF baseline
BaseCascadeRF = CascadeForest(ProbRandomForestClassifier(),cascade_params_list,k_fold=3)
BaseCascadeRF.fit(X_train[:train_size], y_train[:train_size])
y_pre_staged = BaseCascadeRF.predict_staged(X_test)
test_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,y_test), 1, y_pre_staged)
print('\n'.join('level {}, test accuracy: {}'.format(i+1,test_accuracy_staged[i]) for i in range(len(test_accuracy_staged))))

# RF baseline
RF = RandomForestClassifier(n_estimators=1000)
RF.fit(X_train[:train_size], y_train[:train_size])
y_pre = RF.predict(X_test)
print(calc_accuracy(y_pre,y_test))








