# -*- coding: utf-8 -*-
"""
http://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html
Created on Fri Sep 14 16:13:06 2018

@author: Akitaka
"""

import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading
label_prop_model = LabelSpreading()
iris = datasets.load_iris()
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
labels = np.copy(iris.target)
labels[random_unlabeled_points] = -1
label_prop_model.fit(iris.data, labels)
print(labels)
print(iris.target)
print(label_prop_model.transduction_)
print(label_prop_model.predict(iris.data))
