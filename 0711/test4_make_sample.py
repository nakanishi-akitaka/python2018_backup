# -*- coding: utf-8 -*-
"""
scikit-learnを用いたサンプルデータ生成
http://overlap.hatenablog.jp/entry/2015/10/08/022246

Created on Wed Jul 11 15:25:41 2018

@author: Akitaka
"""

### classification sample
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

# サンプルデータの生成
# 1000 samples、5(infomative) + 2(redundant) + 13(independent) =  20 feature のデータを生成
dat = make_classification(n_samples=1000, n_features=20, n_informative=5,
                          n_redundant=2, n_classes=2, n_clusters_per_class=10)

X = dat[0]
y = dat[1]
print("X shape", X.shape)
print("y shape", y.shape)

# 学習用とテスト用データの分割
# 80%を学習、20%をテストに利用する
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 学習モデルの構築とパフォーマンス評価
# ロジスティック回帰、ランダムフォレスト、KNNの３つのモデルを作成しそれぞれのAUCを計算
clf = LogisticRegression()
clf.fit(X_train, y_train)
print("LogisticRegression AUC =", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

clf = RandomForestClassifier(n_estimators=500, random_state=123)
clf.fit(X_train, y_train)
print("RandomForestClassifier AUC =", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)
print("KNeighborsClassifier AUC =", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))