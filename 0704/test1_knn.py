# -*- coding: utf-8 -*-
"""
Python: k 近傍法を実装してみる
https://blog.amedama.jp/entry/2017/03/18/140238

Created on Wed Jul  4 13:45:53 2018

@author: Akitaka
"""

#k 近傍法 (k-Nearest Neighbor algorithm) というのは、
#機械学習において教師あり学習で分類問題を解くためのアルゴリズム。 
#教師あり学習における分類問題というのは、あらかじめ教師信号として
#特徴ベクトルと正解ラベルが与えられるものをいう。 
#その教師信号を元に、未知の特徴ベクトルが与えられたときに正解ラベルを予想しましょう、というもの。
#
#k 近傍法は機械学習アルゴリズムの中でも特にシンプルな実装になっている。
#じゃあ、シンプルな分だけ性能が悪いかというと、そんなことはない。 
#分類精度であれば、他のアルゴリズムに比べても引けを取らないと言われている。 
#ただし、計算量が多いという重大な欠点がある。
#そのため、それを軽減するための改良アルゴリズムも数多く提案されている。
#
#k 近傍法では、与えられた未知の特徴ベクトルを、
#近い場所にある教師信号の正解ラベルを使って分類する。 
#特徴ベクトルで近くにあるものは似たような性質を持っているはず、という考え方になっている。 
#今回は、そんな k 近傍法の基本的な実装を Python で書いてみることにした。
#
#使った環境は次の通り。
#$ sw_vers 
#ProductName:    Mac OS X
#ProductVersion: 10.12.3
#BuildVersion:   16D32
#$ python --version
#Python 3.5.3
#依存パッケージをインストールする
#あらかじめ、今回のソースコードで使う依存パッケージをインストールしておく。
#
#$ pip install numpy scipy scikit-learn


#最近傍法を実装してみる
#k 近傍法では、未知の特徴ベクトルの近くにある k 点の教師信号を用いる。 
#この k 点を 1 にしたときのことを特に最近傍法 (Nearest Neighbor algorithm) と呼ぶ。 
#一番近い場所にある教師信号の正解ラベルを返すだけなので、さらに実装しやすい。 
#そこで、まずは最近傍法から書いてみることにしよう。
#
#次のサンプルコードでは最近傍法を NearestNeighbors というクラスで実装している。 
#インターフェースは scikit-learn っぽくしてみた。
#分類するデータセットは Iris (あやめ) を使っている。

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
        #判別する点と教師データとのユークリッド距離を計算する
        distances = np.array([self._distance(p, x) for p in self._train_data])
        #最もユークリッド距離の近い要素のインデックスを得る
        nearest_index = distances.argmin()
        #最も近い要素のラベルを返す
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
    
    # LOO法で汎化性能を調べる
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


# 上記のサンプルコードでは Leave-One-Out 法というやり方で交差検証をしている。
#
# 交差検証というのは、学習に使わなかったデータを使って正解を導くことができたか調べる方法を指す。 
# モデルの性能は、未知のデータに対する対処能力で比べる必要がある。
# この、未知のデータに対する対処能力のことを汎化性能と呼ぶ。 
# 交差検証をすることで、この汎化性能を測ることができる。
#
# Leave-One-Out 法では、教師信号の中から検証用のデータをあらかじめ一つだけ抜き出しておく。 
# そして、それをモデルが正解できるのか調べるやり方だ。
# 抜き出す対象を一つずつずらしながら、データセットに含まれる要素の数だけ繰り返す。
# 他の交差検証に比べると計算量は増えるものの、厳密で分かりやすい。
#
# 上記のサンプルコードの実行結果は次の通り。
#
# $ python nn.py 
# 0.96
# 汎化性能で 96% の正解率が得られた。 
 
 