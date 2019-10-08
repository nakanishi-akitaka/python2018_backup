# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:11:15 2018

@author: Akitaka
"""
from time                    import time
start = time()

from sklearn.datasets        import fetch_mldata
mnist = fetch_mldata('MNIST original')
print(type(mnist),mnist.data.shape,mnist.target.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, 
                                                    test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#X_train = X_train[:1000]
#y_train = y_train[:1000]
#X_test = X_test[:1000]
#y_test = y_test[:1000]
# Accuracy: 0.858
# F1 score: 0.8568079628847619
# 5.25 seconds

#X_train = X_train[:10000]
#y_train = y_train[:10000]
#X_test = X_test[:10000]
#y_test = y_test[:10000]
# Accuracy: 0.9462
# F1 score: 0.9460479311163679
# 408.24 seconds

#X_train = X_train[:20000]
#y_train = y_train[:20000]
# Accuracy: 0.9577142857142857
# F1 score: 0.9575987020766779
# 956.93 seconds 

#X_train = X_train[:40000]
#y_train = y_train[:40000]
# Accuracy: 0.9678571428571429
# F1 score: 0.9678009137979735
# 1564.92 seconds 

# full
# Accuracy: 0.9717142857142858
# F1 score: 0.9716759551308176
# 2061.73 seconds 
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

from sklearn.neighbors       import KNeighborsClassifier
knc = KNeighborsClassifier()

knc.fit(X_train, y_train)

y_pred = knc.predict(X_test)

from sklearn.metrics         import accuracy_score, f1_score
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred, average='weighted'))

print('{:.2f} seconds '.format(time() - start))
