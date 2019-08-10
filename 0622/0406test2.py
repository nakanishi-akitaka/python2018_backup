#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
iris = datasets.load_iris()
# print(iris['feature_names'])
# print(iris['target_names'])
# print(iris['data'])
# print(iris['target'])

from sklearn.model_selection import train_test_split
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)

print('clf.estimators_')
print( clf.estimators_)
print('')
print('clf.feature_importances_')
print( clf.feature_importances_)
print('')
print('clf.n_features_')
print( clf.n_features_)
print('')
print('clf.n_outputs_')
print( clf.n_outputs_)

from sklearn import metrics
#### Ref
#### http://scikit-learn.org/stable/modules/model_evaluation.html
#### https://qiita.com/nazoking@github/items/958426da6448d74279c7	
# 3.3.2.2
print('metrics.accuracy_score')
print( metrics.accuracy_score(y_test, y_predicted))
# 3.3.2.3
print('metrics.cohen_kappa_score')
print( metrics.cohen_kappa_score(y_test, y_predicted))
# 3.3.2.4
print('metrics.confusion_matrix')
print( metrics.confusion_matrix(y_test, y_predicted))
# 3.3.2.5
print('metrics.classification_report')
print( metrics.classification_report(y_test, y_predicted))
# 3.3.2.6
print('metrics.hamming_loss')
print( metrics.hamming_loss(y_test, y_predicted))
# 3.3.2.7
print('metrics.jaccard_similarity_score')
print( metrics.jaccard_similarity_score(y_test, y_predicted))
# 3.3.2.8
print('metrics.f1_score')
print( metrics.f1_score(y_test, y_predicted, average='weighted'))
print('metrics.precision_score')
print( metrics.precision_score(y_test, y_predicted, average='micro'))
print('metrics.recall_score')
print( metrics.recall_score(y_test, y_predicted, average='micro'))

# 3.3.2.8.1 -> skip
# 3.3.2.8.2
print('metrics.fbeta_score, beta=0.5')
print( metrics.fbeta_score(y_test, y_predicted, average='micro', beta=0.5))
print('metrics.fbeta_score, beta=1')
print( metrics.fbeta_score(y_test, y_predicted, average='micro', beta=1))
print('metrics.fbeta_score, beta=2')
print( metrics.fbeta_score(y_test, y_predicted, average='micro', beta=2))
print('metrics.precision_score,                     average= micro')
print( metrics.precision_score(y_test, y_predicted, average='micro'))
print('metrics.precision_score,                     average= macro')
print( metrics.precision_score(y_test, y_predicted, average='macro'))
print('metrics.precision_score,                     average= weighted')
print( metrics.precision_score(y_test, y_predicted, average='weighted'))

# 3.3.2.9 -> skip
