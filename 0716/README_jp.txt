# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 09:57:47 2018

@author: Akitaka
"""

[1d1]
r2・RMSE・MAE・実測値 vs. 推定値プロット以外の、
回帰分析結果・回帰モデルによる推定結果の評価方法
http://univprof.com/archives/16-07-20-4857140.html
r2・RMSE・MAE と 実測値 vs. 推定値プロットを必ず作る！

example:
test0.py の回帰部分のグラフをyy-plotに変更した
→
分離する
test0_rgr.py
test0_cls.py

推定誤差が中心0の正規分布に従うか？も調べるため、
train, testの推定誤差のヒストグラムを表示させる機能を追加


[1a2] データセットの可視化・見える化
example:
test0.py の分類部分に、X,yの散布図を表示する機能と追加
それに伴い、load_digitsからmake_classificationに変更

ref:
Robust linear model estimation using RANSAC
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html
example:
test1_RANSAC.py

The Iris Dataset
http://scikit-learn.org/0.18/auto_examples/datasets/plot_iris_dataset.html
example:
test2_iris.py

http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
example:
test3_clf.py
