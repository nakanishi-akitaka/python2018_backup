# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:25:03 2018

@author: Akitaka
"""

[1d] [todo]->[done]
先行研究の、XGBoostを水素化物Tcでやってみる
XGBoostでなく、GBoostなら、sklearnにもある！
ref:06/29
06/29はよく読み直す
XGBoostもpythonのコードはあるが、自分でインストールする必要あり
scikit-learn風に作られているらしい
[todo] インストールしてテスト
    ref:
    http://yag.xyz/blog/2015/08/08/xgboost-python/
    http://wolfin.hatenablog.com/entry/2018/02/08/092124
    http://tekenuko.hatenablog.com/entry/2016/09/22/220814
    https://qiita.com/TomokIshii/items/290adc16e2ca5032ca07


http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

サンプルから
Gradient Boosting regression
http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

Partial Dependence Plots
http://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html

Prediction Intervals for Gradient Boosting Regression
http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
Prediction Intervals = 予測区間（よそくくかん）
とは統計学用語で、母集団を仮定した上で、将来観察されるであろう標本値（現在は測定できない）
に対して「どの範囲にあると予測されるか」を示すものである。

Model Complexity Influence
http://scikit-learn.org/stable/auto_examples/applications/plot_model_complexity_influence.html

Tc計算やってみた
ほぼデフォルトパラメーター
最適化したのは'n_estimators':[10, 50, 100, 200, 500]
    上の"Model Complexity..."のサンプルプログラムを参考にした

結果:
Best parameters set found on development set:
{'model__n_estimators': 500}
C:  RMSE, MAE, R^2 = 7.657, 5.292, 0.962
CV: RMSE, MAE, R^2 = 16.192, 10.897, 0.831
P:  RMSE, MAE, R^2 = 43.418, 33.135, 0.000

Predicted Tc is written in file Tc_GBoost_AD_DCV.csv

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 22.909 (+/-1.138)
MAE  DCV: 15.191 (+/-0.498)
R^2  DCV: 0.660 (+/-0.034)

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 33.559 (+/-0.601)
MAE: 26.176 (+/-0.433)
R^2: 0.272 (+/-0.026)
101.51 seconds 

Tcが高い候補 top 10
CaH9,400,189,1
ScH8,400,189,1
ScH9,400,189,1
VH9,400,189,1
VH8,400,189,1
TiH8,400,189,1
CaH8,400,189,1
TiH9,400,189,1
TiH8,500,188,1
VH8,500,188,1

似たようなTcなのは、RFと同様
しかし、今までの(kNN, SVR, RF)と異なり、XH3ではない。
何が原因？
