# -*- coding: utf-8 -*-
"""
https://blog.amedama.jp/entry/2017/12/09/142655
Created on Mon Jun 25 11:55:32 2018

@author: Akitaka
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import random
from sklearn import datasets


def main():
    dataset = datasets.load_digits()
    X = dataset.data
    y = dataset.target

    # データの中から 25 点を無作為に選び出す
    sample_indexes = random.choice(np.arange(len(X)), 25, replace=False)

    # 選んだデータとラベルを matplotlib で表示する
    samples = np.array(list(zip(X, y)))[sample_indexes]
    for index, (data, label) in enumerate(samples):
        # 画像データを 5x5 の格子状に配置する
        plt.subplot(5, 5, index + 1)
        # 軸に関する表示はいらない
        plt.axis('off')
        # データを 8x8 のグレースケール画像として表示する
        plt.imshow(data.reshape(8, 8), cmap=cm.gray_r, interpolation='nearest')
        # 画像データのタイトルに正解ラベルを表示する
        plt.title(label, color='red')
    # グラフを表示する
    plt.show()


if __name__ == '__main__':
    main()