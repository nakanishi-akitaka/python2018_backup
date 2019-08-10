# -*- coding: utf-8 -*-
"""
Machine learning energy gap of ABO2 (classification)
ref:
http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_probas.html

Created on Fri Jun 22 14:15:58 2018

@author: Akitaka
"""

print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=123)
clf2 = RandomForestClassifier(random_state=123)
clf3 = GaussianNB()
# read data from csv file
name = 'test1_cls.csv'
data = np.array(pd.read_csv(name))[:,:]
y=data[:,8]
X=data[:,0:2]

# 0406test2.py

from sklearn.tree import DecisionTreeClassifier
# clf = RandomForestClassifier()
clf = DecisionTreeClassifier()
clf.fit(X, y)
y_pred = clf.predict(X)
if(False):
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
print( metrics.accuracy_score(y, y_pred))
# 3.3.2.3
print('metrics.cohen_kappa_score')
print( metrics.cohen_kappa_score(y, y_pred))
# 3.3.2.4
print('metrics.confusion_matrix')
print( metrics.confusion_matrix(y, y_pred))
# 3.3.2.5
print('metrics.classification_report')
print( metrics.classification_report(y, y_pred))
# 3.3.2.6
print('metrics.hamming_loss')
print( metrics.hamming_loss(y, y_pred))
# 3.3.2.7
print('metrics.jaccard_similarity_score')
print( metrics.jaccard_similarity_score(y, y_pred))
# 3.3.2.8
print('metrics.f1_score')
print( metrics.f1_score(y, y_pred, average='weighted'))
print('metrics.precision_score')
print( metrics.precision_score(y, y_pred, average='micro'))
print('metrics.recall_score')
print( metrics.recall_score(y, y_pred, average='micro'))

# 3.3.2.8.1 -> skip
# 3.3.2.8.2
print('metrics.fbeta_score, beta=0.5')
print( metrics.fbeta_score(y, y_pred, average='micro', beta=0.5))
print('metrics.fbeta_score, beta=1')
print( metrics.fbeta_score(y, y_pred, average='micro', beta=1))
print('metrics.fbeta_score, beta=2')
print( metrics.fbeta_score(y, y_pred, average='micro', beta=2))
print('metrics.precision_score,                     average= micro')
print( metrics.precision_score(y, y_pred, average='micro'))
print('metrics.precision_score,                     average= macro')
print( metrics.precision_score(y, y_pred, average='macro'))
print('metrics.precision_score,                     average= weighted')
print( metrics.precision_score(y, y_pred, average='weighted'))

# 3.3.2.9 -> skip


# http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_probas.html
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                        voting='soft',
                        weights=[1, 1, 5])

# predict class probabilities for all classifiers
probas = [c.fit(X, y).predict_proba(X) for c in (clf1, clf2, clf3, eclf)]

# get class probabilities for the first sample in the dataset
class1_1 = [pr[0, 0] for pr in probas]
class2_1 = [pr[0, 1] for pr in probas]


# plotting

N = 4  # number of groups
ind = np.arange(N)  # group positions
width = 0.35  # bar width

fig, ax = plt.subplots()

# bars for classifier 1-3
p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
            color='green', edgecolor='k')
p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
            color='lightgreen', edgecolor='k')

# bars for VotingClassifier
p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width,
            color='blue', edgecolor='k')
p4 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width,
            color='steelblue', edgecolor='k')

# plot annotations
plt.axvline(2.8, color='k', linestyle='dashed')
ax.set_xticks(ind + width)
ax.set_xticklabels(['LogisticRegression\nweight 1',
                    'GaussianNB\nweight 1',
                    'RandomForestClassifier\nweight 5',
                    'VotingClassifier\n(average probabilities)'],
                   rotation=40,
                   ha='right')
plt.ylim([0, 1])
plt.title('Class probabilities for sample 1 by different classifiers')
plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
plt.show()