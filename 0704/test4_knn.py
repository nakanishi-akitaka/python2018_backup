# -*- coding: utf-8 -*-
"""
Python: k 近傍法を実装してみる
https://blog.amedama.jp/entry/2017/03/18/140238

Created on Wed Jul  4 14:21:51 2018

@author: Akitaka
"""

# 最適な k を探す
# k 近傍法では、計算に近傍何点を使うか (ようするに k) がハイパーパラメータとなっている。
# ハイパーパラメータというのは、機械学習において人間が調整する必要のあるパラメータのことをいう。
#
# 次は、最適な k を探してみることにする。 といっても、やることは単に総当りで探すだけ。
#
# せっかくならパラメータによる汎化性能の違いを可視化したい。 そこで matplotlib も入れておこう。
#
# $ pip install matplotlib
# $ mkdir -p ~/.matplotlib
# $ cat << 'EOF' > ~/.matplotlib/matplotlibrc
# backend: TkAgg
# EOF
# 次のサンプルコードでは k を 1 ~ 20 の間で調整しながら総当りで汎化性能を計算している。
# データセットごとに最適な k が異なるところを見ておきたいので
# Iris (あやめ) と Digits (数字)で調べることにした。
# 自分で実行するときは、データセットのロード部分にあるコメントアウトを切り替えてほしい。

from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def main():
    # データセットをロードする
    # dataset = datasets.load_digits()
    dataset = datasets.load_iris()

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

