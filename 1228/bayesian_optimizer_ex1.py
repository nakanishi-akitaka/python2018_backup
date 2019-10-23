# -*- coding: utf-8 -*-
"""

http://www.techscore.com/blog/2016/12/20/機械学習のハイパーパラメータ探索-ベイズ最適/

Created on Tue Dec 25 16:17:32 2018

@author: Akitaka
"""
"""
ベイズ最適化(GP-UCB アルゴリズム)による探索.
 
環境
  Python 2.7.12
  scikit-learn==0.18.1
  matplotlib==1.5.3
  numpy==1.11.2
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
 
 
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
 
    fig = plt.figure()
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
    plt.savefig('fig%02d.png' % (i))
 
 
 
   
# アプリケーションエントリポイント
if __name__ == '__main__':
    
    # パラメータの取りうる範囲
    x_grid = np.atleast_2d(np.linspace(0, 10, 1001)[:1000]).T
    
    # 初期値として x=1, 9 の 2 点の探索をしておく.
    X = np.atleast_2d([1., 9.]).T
    y = blackbox_func(X).ravel()
 
    
    # Gaussian Processs Upper Confidence Bound (GP-UCB)アルゴリズム
    # --> 収束するまで繰り返す(収束条件などチューニングポイント)
    n_iteration = 13
    for i in range(n_iteration):
    
        # 既に分かっている値でガウス過程フィッティング
        # --> カーネル関数やパラメータはデフォルトにしています(チューニングポイント)
        gp = GaussianProcessRegressor()
        gp.fit(X, y)
        
        # 事後分布が求まる
        posterior_mean, posterior_sig = gp.predict(x_grid, return_std=True)
        
        # 目的関数を最大化する x を次のパラメータとして選択する
        # --> βを大きくすると探索重視(初期は大きくし探索重視しイテレーションに同期して減衰させ活用を重視させるなど、チューニングポイント)
        idx = acq_ucb(posterior_mean, posterior_sig, beta=100.0)
        x_next = x_grid[idx]
    
        plot(x_grid, y, X, posterior_mean, posterior_sig, title='Iteration=%2d,  x_next = %f'%(i+2, x_next))
    
        # 更新
        X = np.atleast_2d([np.r_[X[:, 0], x_next]]).T
        y = np.r_[y, blackbox_func(x_next)]
        
    
    print("Max x=%f" % (x_next))



