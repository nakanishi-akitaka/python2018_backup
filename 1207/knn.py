# -*- coding: utf-8 -*-
"""
https://blog.amedama.jp/entry/2017/03/18/140238
Created on Thu Dec  6 15:58:10 2018

@author: Akitaka
"""
from collections import Counter

import numpy as np

from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors       import NearestNeighbors


class KNearestNeighbors(object):

    def __init__(self, k=1):
        self._train_data = None
        self._target_data = None
        self._k = k

    def fit(self, train_data, target_data):
        """訓練データを学習する"""
        # あらかじめ計算しておけるものが特にないので保存だけする
        self._train_data = train_data
        self._target_data = target_data

    def predict(self, x):
        """訓練データから予測する"""
        # 判別する点と教師データとのユークリッド距離を計算する
        distances = np.array([self._distance(p, x) for p in self._train_data])
        # ユークリッド距離の近い順でソートしたインデックスを得る
        nearest_indexes = distances.argsort()[:self._k]
        # 最も近い要素のラベルを返す
        nearest_labels = self._target_data[nearest_indexes]
        # 近傍のラベルで一番多いものを予測結果として返す
        c = Counter(nearest_labels)
        return c.most_common(1)[0][0]

    def _distance(self, p0, p1):
        """二点間のユークリッド距離を計算する"""
        return np.sum((p0 - p1) ** 2)

class KNearestNeighbors_Inheritance(BaseEstimator, RegressorMixin):

    def __init__(self, n_neighbors=5):
        self._train_data = None
        self._target_data = None
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self._train_data = X
        self._target_data = y

    def predict(self, x):
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        neigh.fit(self._train_data)
        neigh_dist, neigh_ind = neigh.kneighbors(x)
        y_pred = np.mean(self._target_data[neigh_ind], axis=1)
#        if self._target_data.ndim == 1:
#            y_pred = y_pred.ravel()
        return y_pred

def main():
    iris_dataset = datasets.load_iris()

    features = iris_dataset.data
    targets = iris_dataset.target

    predicted_labels = []

    loo = LeaveOneOut()
    for train, test in loo.split(features):
        train_data = features[train]
        target_data = targets[train]

        model = KNearestNeighbors(k=3)
        model.fit(train_data, target_data)

        predicted_label = model.predict(features[test])
        predicted_labels.append(predicted_label)

    score = accuracy_score(targets, predicted_labels)
    print(score)


if __name__ == '__main__':
    main()
