# -*- coding: utf-8 -*-
"""
Python: k 近傍法を実装してみる
https://blog.amedama.jp/entry/2017/03/18/140238

Created on Wed Jul  4 14:08:38 2018

@author: Akitaka
"""

# k 近傍法を実装してみる
# 先ほど示した最近傍法の実装では、最寄りの教師信号だけを使うものとなっていた。
# 今度は、より汎用的に近くにある k 点の教師信号を使う実装にしてみる。
#
# 次のサンプルコードでは KNearestNeighbors クラスのコンストラクタに k を渡せるようになっている。
# 実装としては、分類するときに教師信号をユークリッド距離でソートした上で k 個を取り出している。
# ひとまず k については 3 を指定した。 もしこれを 1 にすれば最近傍法になる。

from collections import Counter

import numpy as np

from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

class kNearestNeighbors(object):
    
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
        #判別する点と教師データとのユークリッド距離を計算する
        distances = np.array([self._distance(p, x) for p in self._train_data])
        #ユークリッド距離の近い順でソートしたインデックスを得る
        nearest_indexes = distances.argsort()[:self._k]
        #最も近い要素のラベルを返す
        nearest_labels = self._target_data[nearest_indexes]
        # 近傍のラベルで一番多いものを予測結果として返す
        c = Counter(nearest_labels)
        return c.most_common(1)[0][0]
    
    def _distance(self, p0, p1):
        """二点間のユークリッド距離を計算する"""
        return np.sum((p0 - p1) ** 2)
    
def main():
    # Iris データセットをロードする
    iris_dataset = datasets.load_iris()
    
    # 特徴データとラベルデータを取り出す
    features = iris_dataset.data
    targets = iris_dataset.target
    
    # LOO法で汎化性能を調べる
    predicted_labels = []
    
    loo = LeaveOneOut()
    for train, test in loo.split(features):
        train_data = features[train]
        target_data = targets[train]
        
        # モデルを学習させる
        model = kNearestNeighbors(k=3)
        model.fit(train_data, target_data)
        
        # 一つ抜いたテストデータを識別させる
        predicted_label = model.predict(features[test])
        predicted_labels.append(predicted_label)
        
    # 正解率を出力する
    score = accuracy_score(targets, predicted_labels)
    print(score)
    
if __name__ == '__main__':
    main()

# 上記の実行結果は次の通り。
#
# $ python knn.py
# 0.96
# 汎化性能は k=1 のときと変わらないようだ。

