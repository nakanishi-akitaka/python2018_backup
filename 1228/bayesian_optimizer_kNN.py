# -*- coding: utf-8 -*-
"""
ベイズ最適化(GP-UCB アルゴリズム)による探索.
環境
  Python 2.7.12
  scikit-learn==0.18.1
  matplotlib==1.5.3
  numpy==1.11.2

http://www.techscore.com/blog/2016/12/20/機械学習のハイパーパラメータ探索-ベイズ最適/

Created on Fri Dec 28 15:32:12 2018

@author: Akitaka
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors                import KNeighborsRegressor
from sklearn.neighbors                import NearestNeighbors

 
def blackbox_func(x):
    """
    ブラックボックス関数（例題なので x sin(x) としています）
    --> 本来はモデル学習
        ex) y = svm(学習データ, x(ハイパーパラメータ)) の結果など最大化したい値を返す.
    """
    return x * np.sin(x)
 
 
def acq_ucb(mean, sig, beta=3):
    """
    獲得関数 (Upper Confidence Bound)
    $ x_t = argmax\ \mu_{t-1} + \sqrt{\beta_t} \sigma_{t-1}(x) $
    """
    return np.argmax(mean + sig * np.sqrt(beta))
 
 
def plot(x, y, X, y_pred, sigma, title=""):
    plt.figure(figsize=(6,3))
    plt.plot(x, blackbox_func(x), 'r:', label=u'$blackbox func(x) = x\,\sin(x)$')
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.96 * sigma,(y_pred + 1.96 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.ylim(-10, 20)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()
 


def dist_knn(X_train, X_test, n_neighbors=2):
    """
    Determination of distance by k-Nearest Neighbor 

    Parameters
    ----------
    X_train : array-like, shape = [n_samples, n_features]
        X training data

    X_test : array-like, shape = [n_samples, n_features]
        X test data

    n_neighbors : integer
        number of neighbers

    Returns
    -------
    array-like, shape = [n_samples]
        average of distance between X_test and k-Nearest Neighbor 
    """
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X_train)
    dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
    return dist

# アプリケーションエントリポイント
if __name__ == '__main__':
    
    # パラメータの取りうる範囲
    x_grid = np.atleast_2d(np.linspace(0, 10, 1001)[:1000]).T
    
    # 初期値として x=1, 9 の 2 点の探索をしておく.
    X = np.atleast_2d([1., 9.]).T
    y = blackbox_func(X).ravel()
 
    
    # Gaussian Processs Upper Confidence Bound (GP-UCB)アルゴリズム
    # --> 収束するまで繰り返す(収束条件などチューニングポイント)
    n_iteration = 50
    for i in range(n_iteration):
    
        # 既に分かっている値でガウス過程フィッティング
        # --> カーネル関数やパラメータはデフォルトにしています(チューニングポイント)
        gp = GaussianProcessRegressor()
        gp = KNeighborsRegressor(n_neighbors=2)
        gp.fit(X, y)

        # 事後分布が求まる
        posterior_mean = gp.predict(x_grid)
#        posterior_sig = dist_knn(X, x_grid)
        posterior_sig = dist_knn(X, x_grid, min([i+1,5]))

        # 目的関数を最大化する x を次のパラメータとして選択する
        # --> βを大きくすると探索重視(初期は大きくし探索重視しイテレーションに同期して減衰させ活用を重視させるなど、チューニングポイント)
        idx = acq_ucb(posterior_mean, posterior_sig, beta=100.0)
        x_next = x_grid[idx]
    
        plot(x_grid, y, X, posterior_mean, posterior_sig, title='Iteration=%2d,  x_next = %f'%(i+2, x_next))
    
        # 更新
        X = np.atleast_2d([np.r_[X[:, 0], x_next]]).T
        y = np.r_[y, blackbox_func(x_next)]
        
    
    print("Max x=%f" % (x_next))



