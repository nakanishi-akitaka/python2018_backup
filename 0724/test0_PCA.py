# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:16:42 2018

@author: Akitaka
"""

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_) 

pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)                 
print(pca.explained_variance_ratio_) 


pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)
print(pca.explained_variance_ratio_) 
