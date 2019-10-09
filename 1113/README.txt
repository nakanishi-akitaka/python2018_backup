# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:40:07 2018

@author: Akitaka
"""

[1b] 水素化物Tc計算
[1b1] 更新したデータベースを使用して７つのモデルで計算

Tc_to_csv.pyでデータベースをから変換
！三元系の水素化物は除去して、エラーを回避する

前回との比較
C:\Users\Akitaka\Downloads\python\1026\Tc_4model_AD_DCV.py
20181026:EN, RR, LASSO, kNN

C:\Users\Akitaka\Downloads\python\1105\Tc_6model_AD_DCV.py
20181105: ...+RF, SVR

Ridge Regression
{'model__alpha': 1.0}
C:  RMSE, MAE, R^2 = 33.894, 25.541, 0.257
CV: RMSE, MAE, R^2 = 34.458, 25.933, 0.233
P:  RMSE, MAE, R^2 = 42.513, 35.734, 0.000
DCV:RMSE, MAE, R^2 = 34.470 (+/-0.307), 25.939 (+/-0.215), 0.232 (+/-0.014)
y-r:RMSE, MAE, R^2 = 39.067 (+/-0.125), 30.873 (+/-0.097), 0.013 (+/-0.006)
→
{'model__alpha': 1.0}
C:  RMSE, MAE, R^2 = 47.856, 34.911, 0.310
CV: RMSE, MAE, R^2 = 48.292, 35.298, 0.297
P:  RMSE, MAE, R^2 = 62.263, 53.718, 0.000
DCV:RMSE, MAE, R^2 = 48.635 (+/-0.335), 35.465 (+/-0.278), 0.287 (+/-0.010)
y-r:RMSE, MAE, R^2 = 57.334 (+/-0.149), 42.207 (+/-0.156), 0.010 (+/-0.005)

formula,P,Tc,AD
NaH9,500,120,1
VH9,500,119,1
AlH9,500,119,1
CrH9,500,119,1
SiH9,500,119,1
TiH9,500,119,1
ScH9,500,119,1
MgH9,500,119,1
KH9,500,119,1


LASSO
{'model__alpha': 0.6000000000000001}
C:  RMSE, MAE, R^2 = 33.918, 25.422, 0.256
CV: RMSE, MAE, R^2 = 34.359, 25.741, 0.237
P:  RMSE, MAE, R^2 = 39.666, 33.257, 0.000
DCV:RMSE, MAE, R^2 = 34.662 (+/-0.317), 25.966 (+/-0.239), 0.223 (+/-0.014)
y-r:RMSE, MAE, R^2 = 39.062 (+/-0.173), 30.923 (+/-0.092), 0.014 (+/-0.009)
→
{'model__alpha': 0.1}
C:  RMSE, MAE, R^2 = 47.857, 34.907, 0.310
CV: RMSE, MAE, R^2 = 48.941, 35.755, 0.278
P:  RMSE, MAE, R^2 = 62.730, 54.278, 0.000
DCV:RMSE, MAE, R^2 = 49.141 (+/-0.719), 35.622 (+/-0.360), 0.272 (+/-0.021)
y-r:RMSE, MAE, R^2 = 57.340 (+/-0.189), 42.177 (+/-0.162), 0.009 (+/-0.007)

formula,P,Tc,AD
H9Cl,500,119,1
AlH9,500,119,1
KH9,500,119,1
ScH9,500,119,1
CaH9,500,119,1
MgH9,500,119,1
H9S,500,119,1
NaH9,500,119,1
PH9,500,119,1


Elastic Net
{'model__alpha': 0.1, 'model__l1_ratio': 0.5}
C:  RMSE, MAE, R^2 = 33.913, 25.446, 0.257
CV: RMSE, MAE, R^2 = 34.304, 25.680, 0.239
P:  RMSE, MAE, R^2 = 39.919, 33.522, 0.000
DCV:RMSE, MAE, R^2 = 34.514 (+/-0.293), 25.776 (+/-0.178), 0.230 (+/-0.013)
y-r:RMSE, MAE, R^2 = 39.140 (+/-0.078), 30.843 (+/-0.117), 0.010 (+/-0.004)
→
{'model__alpha': 0.1, 'model__l1_ratio': 0.30000000000000004}
C:  RMSE, MAE, R^2 = 47.946, 35.136, 0.307
CV: RMSE, MAE, R^2 = 48.397, 35.548, 0.294
P:  RMSE, MAE, R^2 = 62.469, 55.259, 0.000
DCV:RMSE, MAE, R^2 = 48.532 (+/-0.253), 35.501 (+/-0.156), 0.290 (+/-0.007)
y-r:RMSE, MAE, R^2 = 57.333 (+/-0.168), 42.184 (+/-0.156), 0.010 (+/-0.006)

formula,P,Tc,AD
NaH9,500,115,1
MgH9,500,115,1
AlH9,500,115,1
CaH9,500,114,1
FeH9,500,114,1
TiH9,500,114,1
VH9,500,114,1
H9Cl,500,114,1
ScH9,500,114,1



kNN
{'model__n_neighbors': 2}
C:  RMSE, MAE, R^2 =  9.528,  5.602, 0.941
CV: RMSE, MAE, R^2 = 23.373, 14.106, 0.647
P:  RMSE, MAE, R^2 = 34.608, 21.794, 0.000
DCV:RMSE, MAE, R^2 = 37.623 (+/-1.594), 26.272 (+/-1.263), 0.083 (+/-0.077)
y-r:RMSE, MAE, R^2 = 37.263 (+/-0.240), 29.070 (+/-0.389), 0.102 (+/-0.012)
→
{'model__n_neighbors': 1}
C:  RMSE, MAE, R^2 =  9.043,  4.504, 0.975
CV: RMSE, MAE, R^2 = 19.990, 12.751, 0.880
P:  RMSE, MAE, R^2 = 54.843, 32.129, 0.000
DCV:RMSE, MAE, R^2 = 53.343 (+/-1.449), 34.661 (+/-1.111), 0.142 (+/-0.047)
y-r:RMSE, MAE, R^2 = 54.553 (+/-0.607), 40.170 (+/-0.413), 0.103 (+/-0.020)

formula,P,Tc,AD
YH9,350,310,1
In2H9,400,310,1
CdH9,400,310,1
YH9,400,310,1
H9Rh,400,310,1
H9Rh,350,310,1
NbH9,350,310,1
H9Ru2,350,310,1
H9Pd,350,310,1



Random Forest
{'model__max_features': 0.8}
C:  RMSE, MAE, R^2 =  8.236,  5.620, 0.956
CV: RMSE, MAE, R^2 = 19.062, 13.328, 0.765
P:  RMSE, MAE, R^2 = 46.904, 39.253, 0.000
DCV:RMSE, MAE, R^2 = 21.343 (+/-1.253), 13.945 (+/-0.650), 0.705 (+/-0.035)
y-r:RMSE, MAE, R^2 = 28.118 (+/-1.285), 20.716 (+/-0.618), 0.488 (+/-0.047)
→
{'model__max_features': 0.30000000000000004}
C:  RMSE, MAE, R^2 = 11.193,  7.720, 0.962
CV: RMSE, MAE, R^2 = 24.986, 16.730, 0.812
P:  RMSE, MAE, R^2 = 51.537, 42.050, 0.000
DCV:RMSE, MAE, R^2 = 29.303 (+/-1.209), 18.499 (+/-0.646), 0.741 (+/-0.021)
y-r:RMSE, MAE, R^2 = 41.138 (+/-1.067), 28.698 (+/-0.750), 0.490 (+/-0.026)

formula,P,Tc,AD
RbH9,150,235,1
SrH9,150,235,1
ZrH9,150,235,1
YH9,150,235,1
H9Ru,150,216,1
NbH9,150,216,1
TcH9,150,216,1
MoH9,150,216,1
AgH9,150,212,1




SVR
{'model__C': 1024.0, 'model__epsilon': 1.0, 'model__gamma': 1024.0}
C:  RMSE, MAE, R^2 =  5.854,  3.615, 0.978
CV: RMSE, MAE, R^2 = 25.132, 16.377, 0.592
P:  RMSE, MAE, R^2 = 45.290, 45.252, 0.000
DCV:RMSE, MAE, R^2 = 29.284 (+/-1.447), 19.486 (+/-0.684), 0.444 (+/-0.054)
y-r:RMSE, MAE, R^2 = 28.158 (+/-1.541), 15.835 (+/-1.129), 0.486 (+/-0.058)
→
{'model__C': 1024.0, 'model__epsilon': 1.0, 'model__gamma': 1024.0}
C:  RMSE, MAE, R^2 =  6.875,  4.139, 0.986
CV: RMSE, MAE, R^2 = 28.946, 19.218, 0.748
P:  RMSE, MAE, R^2 = 58.477, 58.437, 0.000
DCV:RMSE, MAE, R^2 = 38.653 (+/-3.118), 24.732 (+/-1.382), 0.547 (+/-0.071)
y-r:RMSE, MAE, R^2 = 43.771 (+/-2.047), 22.681 (+/-1.062), 0.422 (+/-0.053)

formula,P,Tc,AD
YH9,150,254,1
ScH7,300,168,1
ScH9,400,168,1
CaH6,150,163,1
ScH9,300,161,1
LaH6,100,155,1
AsH8,450,150,1
ScH8,400,149,1
AsH8,400,142,1



GPR
C:  RMSE, MAE, R^2 = 5.108, 3.285, 0.983
→
C:  RMSE, MAE, R^2 = 5.875, 3.837, 0.990

formula,P,Tc,std,AD
YH9,150,260,1,1
ScH9,400,174,1,1
ScH7,300,168,0,1
CaH6,150,167,1,1
ScH9,300,162,0,1
LaH6,100,162,1,1
ScH8,400,150,0,1
AsH8,400,143,0,1
AsH8,350,140,0,1



[1b2] 仕様変更
Tc_new.py, tc_new.csvで作成
1.tc.csvを変換(tc_to_csv.py), Tc計算のプログラム(tc_6model_AD_DCV.py)を(ふたたび)統合
2.計算前に予めスケーリングする → DCV, y-randmでもスケーリングした状態になる
3.スケーリングの逆変換から化学式を出すのは色々面倒だしエラーの元なので、化学式も残しておく
4.原子番号や原子の個数以外のパラメータにも対応できるように汎用性を高める

2.によって、[1b1]のDCV結果が変わった！
おそらくはDCVでスケーリングをやっていなかったせい
例 kNN
{'model__n_neighbors': 1}
C:  RMSE, MAE, R^2 =  9.043,  4.504, 0.975
CV: RMSE, MAE, R^2 = 19.990, 12.751, 0.880
P:  RMSE, MAE, R^2 = 54.843, 32.129, 0.000
DCV:RMSE, MAE, R^2 = 53.343 (+/-1.449), 34.661 (+/-1.111), 0.142 (+/-0.047)
y-r:RMSE, MAE, R^2 = 54.553 (+/-0.607), 40.170 (+/-0.413), 0.103 (+/-0.020)
→
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  9.043,  4.504, 0.975
CV: RMSE, MAE, R^2 = 22.845, 14.049, 0.843
P:  RMSE, MAE, R^2 = 58.395, 35.507, 0.000
DCV:RMSE, MAE, R^2 = 30.493 (+/-2.053), 17.645 (+/-1.220), 0.719 (+/-0.037)
y-r:RMSE, MAE, R^2 = 54.101 (+/-0.710), 39.979 (+/-0.729), 0.118 (+/-0.023)

今更気づいたが、以前のtc_test.csvだと、
X9H9 をそのまま原子番号と原子の個数に変換して扱っている
material.reduced_formulaを使えば、X9H9=XHに変換される
また、.drop_duplicates()メソッドで、重複した行を削除できる
20540/63990がダブり

ref:
https://note.nkmk.me/pandas/
https://note.nkmk.me/python-pandas-duplicated-drop-duplicates/
https://note.nkmk.me/python-numpy-dtype-astype/
https://deepage.net/features/numpy-dtype.html
http://publicjournal.hatenablog.com/entry/2017/02/24/234333
https://www.sejuku.net/blog/71841


[1b3] スケーリングの順序について
C:\Users\Akitaka\Downloads\python\1009\test0_tips_jp.txt
<div>
StandardScaler()の位置は、train, test分割の前後どっち？
　→　後！
ref:20180803 [1d] 
X_train　.fit_transform(X_train)
X_test　.fit(X_train) -> .transform(X_test)
でなければならない

プログラムは、以下の通り
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=...)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
パイプラインやPCAを使う場合
    std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
    std_clf.fit(X_train, y_train)
    pred_test_std = std_clf.predict(X_test)
    scaler = std_clf.named_steps['standardscaler']
    X_train_std = pca_std.transform(scaler.transform(X_train))
</div>
とある
二重交差検証の場合は？　内側？外側？どちらでスケーリング？
極端な話、充分にデータ量が多ければ、分割前でも後でも、内側でも外側でも、
平均値も標準偏差もほとんど変わらないので問題はないハズ
