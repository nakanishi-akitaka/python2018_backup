# -*- coding: utf-8 -*-
"""

ref:
https://blog.amedama.jp/entry/2017/12/09/142655
Created on Mon Jun 25 11:48:50 2018

@author: Akitaka
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

def main():
    dataset = datasets.load_digits()

    X = dataset.data
    y = dataset.target

    plt.figure(figsize=(12, 8))

    # 主な多様体学習アルゴリズム (と主成分分析)
    manifolders = {
        'PCA': PCA(),
        'MDS': MDS(),
        'Isomap': Isomap(),
        'LLE': LocallyLinearEmbedding(),
        'Laplacian Eigenmaps': SpectralEmbedding(),
        't-SNE': TSNE(),
    }
    for i, (name, manifolder) in enumerate(manifolders.items()):
        plt.subplot(2, 3, i + 1)

        # 多様体学習アルゴリズムを使って教師データを 2 次元に縮約する
        X_transformed = manifolder.fit_transform(X)

        # 縮約した結果を二次元の散布図にプロットする
        for label in np.unique(y):
            plt.title(name)
            plt.scatter(X_transformed[y == label, 0], X_transformed[y == label, 1])

    plt.show()

if __name__ == '__main__':
    main()