# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:34:20 2018

@author: Akitaka
"""

[1d] SVM+OCSVM
ref:0713
example:
test1_SVM_OCSVM.pyをアップデート
Inlier  sample, number of good/bad predictions: 4 174
Outlier sample, number of good/bad predictions: 0 22
で表示していたのはgood/badが逆！つまりOutlierは全部正解

適用範囲の外のサンプルを白抜きの丸で表示する
→適用範囲の境界線を表示する方が分かりやすい！
＋予測が外れたサンプルには×印をプロットした
＋スケーリング追加

ref:
http://scikit-learn.org/stable/auto_examples/exercises/plot_iris_exercise.html
http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html

