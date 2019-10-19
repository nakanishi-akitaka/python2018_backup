# -*- coding: utf-8 -*-
"""
https://blog.amedama.jp/entry/2017/03/18/140238
Created on Thu Dec  6 15:43:03 2018

@author: Akitaka
"""

import numpy as np

from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score


class NearestNeighbors(object):

    def __init__(self):
        self._train_data = None
        self._target_data = None

    def fit(self, train_data, target_data):
        """訓練データを学習する"""
        # あらかじめ計算しておけるものが特にないので保存だけする
        self._train_data = train_data
        self._target_data = target_data

    def predict(self, x):
        """訓練データから予測する"""
        # 判別する点と教師データとのユークリッド距離を計算する
        distances = np.array([self._distance(p, x) for p in self._train_data])
        # 最もユークリッド距離の近い要素のインデックスを得る
        nearest_index = distances.argmin()
        # 最も近い要素のラベルを返す
        return self._target_data[nearest_index]

    def _distance(self, p0, p1):
        """二点間のユークリッド距離を計算する"""
        return np.sum((p0 - p1) ** 2)


def main():
    # Iris データセットをロードする
    iris_dataset = datasets.load_iris()

    # 特徴データとラベルデータを取り出す
    features = iris_dataset.data
    targets = iris_dataset.target

    # LOO 法で汎化性能を調べる
    predicted_labels = []

    loo = LeaveOneOut()
    for train, test in loo.split(features):
        train_data = features[train]
        target_data = targets[train]

        # モデルを学習させる
        model = NearestNeighbors()
        model.fit(train_data, target_data)
        
        # 一つ抜いたテストデータを識別させる
        predicted_label = model.predict(features[test])
        predicted_labels.append(predicted_label)

    # 正解率を出力する
    score = accuracy_score(targets, predicted_labels)
    print(score)


if __name__ == '__main__':
    main()

