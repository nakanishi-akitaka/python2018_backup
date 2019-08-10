# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:15:52 2018

@author: Akitaka
http://scikit-learn.org/stable/modules/svm.html
"""
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print(clf.predict([[2., 2.]]))

# get support vectors
print(clf.support_vectors_)

# get indices of support vectors
print(clf.support_)

# get number of support vectors for each class
print(clf.n_support_)

# 1.4.1.1. Multi-class classification
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y) 
dec = clf.decision_function([[1]])
print(dec.shape[1]) # 4 classes: 4*3/2 = 6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
print(dec.shape[1]) # 4 classes

lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y) 
dec = lin_clf.decision_function([[1]])
print(dec.shape[1])

# 1.4.2. Regression
from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X, y) 
print(clf.predict([[1, 1]]))

# 1.4.6. Kernel functions
linear_svc = svm.SVC(kernel='linear')
print(linear_svc.kernel)
rbf_svc = svm.SVC(kernel='rbf')
print(rbf_svc.kernel)

# 1.4.6.1. Custom Kernels
import numpy as np
from sklearn import svm
def my_kernel(X, Y):
     return np.dot(X, Y.T)
clf = svm.SVC(kernel=my_kernel)

# 1.4.6.1.2. Using the Gram matrix
from sklearn import svm
X = np.array([[0, 0], [1, 1]])
y = [0, 1]
clf = svm.SVC(kernel='precomputed')
# linear kernel computation
gram = np.dot(X, X.T)
clf.fit(gram, y) 
# predict on training examples
print(clf.predict(gram))
