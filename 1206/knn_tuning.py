# -*- coding: utf-8 -*-
"""
https://blog.amedama.jp/entry/2017/03/18/140238
Created on Thu Dec  6 16:06:36 2018

@author: Akitaka
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def main():
    # データセットをロードする
    dataset = datasets.load_digits()
    # dataset = datasets.load_iris()

    # 特徴データとラベルデータを取り出す
    features = dataset.data
    targets = dataset.target

    # 検証する近傍数の上限
    K = 20
    ks = range(1, K + 1)

    # 使う近傍数ごとに正解率を計算する
    accuracy_scores = []
    for k in ks:
        # Leave-One-Out 法で汎化性能を測る
        predicted_labels = []
        loo = LeaveOneOut()
        for train, test in loo.split(features):
            train_data = features[train]
            target_data = targets[train]

            # モデルを学習させる    
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(train_data, target_data)
    
            # 一つだけ取り除いたテストデータを識別させる
            predicted_label = model.predict(features[test])
            predicted_labels.append(predicted_label)
    
        # 正解率を計算する
        score = accuracy_score(targets, predicted_labels)
        print('k={0}: {1}'.format(k, score))

        accuracy_scores.append(score)

    # 使う近傍数ごとの正解率を折れ線グラフで可視化する
    X = list(ks)
    plt.plot(X, accuracy_scores)

    plt.xlabel('k')
    plt.ylabel('accuracy rate')
    plt.show()
    

if __name__ == '__main__':
    main()