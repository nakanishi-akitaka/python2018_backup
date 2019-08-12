# -*- coding: utf-8 -*-
"""
Python: k 近傍法を実装してみる
https://blog.amedama.jp/entry/2017/03/18/140238

Created on Wed Jul  4 14:06:06 2018

@author: Akitaka
"""

# scikit-learn を使う場合
# ちなみに、自分で書く代わりに scikit-learn にある実装を使う場合も紹介しておく。
#
# 次のサンプルコードは k近傍法の実装をscikit-learn の KNeighborsClassifier に代えたもの。
# インターフェースを揃えてあったので、使うクラスが違う以外は先ほどと同じソースコードになっている。
# scikit-learn で最近傍法をしたいときは KNeighborsClassifierのkに1を指定するだけで良い。

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def main():
    iris_dataset = datasets.load_iris()

    features = iris_dataset.data
    targets = iris_dataset.target

    predicted_labels = []

    loo = LeaveOneOut()
    for train, test in loo.split(features):
        train_data = features[train]
        target_data = targets[train]

        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(train_data, target_data)

        predicted_label = model.predict(features[test])
        predicted_labels.append(predicted_label)

    score = accuracy_score(targets, predicted_labels)
    print(score)


if __name__ == '__main__':
    main()


# 上記のサンプルコードの実行結果は次の通り。
#
# $ python knn_scikit.py 
# 0.96
# 当然だけど同じ班化性能になっている。
