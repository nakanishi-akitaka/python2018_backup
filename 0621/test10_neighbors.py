# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:37:56 2018

@author: Akitaka
http://scikit-learn.org/stable/modules/neighbors.html
"""

# 1.6. Nearest Neighbors
# 1.6.1.1. Finding the Nearest Neighbors
from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print(indices)
print(distances)
print(nbrs.kneighbors_graph(X).toarray())

# 1.6.1.2. KDTree and BallTree Classes
from sklearn.neighbors import KDTree
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
print(kdt.query(X, k=2, return_distance=False))

# 1.6.5. Nearest Centroid Classifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))
