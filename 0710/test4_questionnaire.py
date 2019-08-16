# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:35:57 2018

@author: Akitaka
"""

#coding: UTF-8
# Here your code !
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

data = ['I artificial intelligence is love. Technology is wonderful.',
        'For machine learning, I believe that the thing to brighten the future.',
        'By learning artificial intelligence, I think that it is possible to increase the revenue.',
        'I think management is better understanding of artificial intelligence, and the cost can be reduced.',
        'In the same way now with the air, the process goes the introduction of artificial intelligence, the world will come in handy.',
        'Artificial intelligence is a devil. Thing to destroy the human race.',
        'AI will take away the employment of people significantly.',
        'Human was defeated in artificial intelligence, human values will drop significantly',
        'Now I want to immediately destroyed. Anna what is unnecessary. human values drop.',
        'What if revenue down heh it is because of the artificial intelligence.'
        ]

data2 = [1,1,1,1,1,0,0,0,0,0]

count = CountVectorizer()
bag = count.fit_transform(data)

# 単語辞書を作成。アンケートデータに出てくる単語を、すべて並べたもの。
print("-----dictionary-----")
print(count.vocabulary_)

print("-----vectr_transform------")
print(bag.toarray())

# tfidfで単語データの重要度を評価（isやIはいろんな文書で使われるので重要でないとか）
tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
np.set_printoptions(precision=2)
data3 = tfidf.fit_transform(count.fit_transform(data)).toarray()

print("-----TF-IDF-------")
print(tfidf.fit_transform(count.fit_transform(data)).toarray())

#モデル作成
estimator = svm.SVC(C=14.0, gamma=0.015, probability=True)
estimator.fit(data3, data2)

print("------交差検定--------")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data3, data2, test_size=0.2, random_state=0)
print(estimator.score(X_test, y_test))