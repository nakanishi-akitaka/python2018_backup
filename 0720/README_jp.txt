# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:38:48 2018

@author: Akitaka
"""

[1c] Tcへの応用
[1c1] チェック
example:
test1_check.py

0426test1.pyと、test2_Tc_SVM.pyとの違いを探る
前者：tc.csvから化学式を読み込む
後者：tc_train.csvから原子番号と個数を読み込む
結局は同じになる！

[1c3] Tc予測
example:
test2_Tc_SVM.py
上記のtest2_Tc_SVM.pyと0426test1.pyを足し合わせたもの

方法
range_c = 2**np.arange(  -5, 11, dtype=float)
range_e = 2**np.arange( -10,  1, dtype=float)
range_g = 2**np.arange( -20, 11, dtype=float)
pipe = Pipeline([('scaler', StandardScaler()),('svr', SVR())])
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
score='neg_mean_absolute_error'
