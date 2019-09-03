# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:58:18 2018

@author: Akitaka
"""

時系列データを扱うときの３つの注意点(ソフトセンサー解析など)[データ解析用のPythonプログラム付き]
https://datachemeng.com/pointsoftimeseriesdataanalysis/
    トレーニングデータが更新される
    目的変数 y の測定には時間がかかり、トレーニングデータ更新のときにその時間を考慮しなければならない
    説明変数 X について、時間的に遅れて y と関係していることがある
→
デモンストレーションを実行
    今回は、こちらの本 (L. Fortuna, S. Graziani, A. Rizzo, M. G. Xibilia,
     Soft sensors for monitoring and control of industrial processes.
     London: Springer-Verlag; 2007) に説明のあるデブタナイザーのデータセットを用いた
    時系列データを解析したPythonプログラムを示します。
    こちらのGitHubの demo_time_series_data_analysis_lwpls.py をご利用ください。
    モデリング手法は、Just-In-Time (JIT) モデリング手法の一つである 
    Locally-Weighted Partial Least Squares (LWPLS) です。
example:
lwpls/demo_time_series_data_analysis_lwpls.py
