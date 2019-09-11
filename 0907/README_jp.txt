# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:35:49 2018

@author: Akitaka
"""


[1c] Tc再計算
C:\Users\Akitaka\Downloads\python\0907\
test2_Tc_kNN_AD_DCV.py
test2_Tc_RF_AD_DCV.py
test2_Tc_SVM_OCSVM_DCV.py

[1c1] kNN
Best parameters set found on development set:
{'model__n_neighbors': 3}
C:  RMSE, MAE, R^2 = 14.488,  8.648, 0.864
CV: RMSE, MAE, R^2 = 26.025, 15.630, 0.562
P:  RMSE, MAE, R^2 = 34.647, 22.440, 0.000

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 37.386 (+/-1.028)
MAE  DCV: 27.360 (+/-1.040)
R^2  DCV: 0.096 (+/-0.049)
20.19 seconds 

Top10
formula,P,Tc,AD
ScH3,250,193,1
VH3,200,193,1
VH3,250,193,1
CaH3,250,193,1
H3S,250,193,1
CrH3,250,193,1
H3Cl,250,193,1
CrH3,200,193,1
KH3,250,193,1
TiH3,250,193,1

[1c2] RF
Best parameters set found on development set:
{'model__max_features': 0.8}
C:  RMSE, MAE, R^2 =  8.780,  5.884, 0.950
CV: RMSE, MAE, R^2 = 18.495, 12.000, 0.779
P:  RMSE, MAE, R^2 = 48.940, 42.101, 0.000

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 22.155 (+/-2.427)
MAE  DCV: 14.443 (+/-1.113)
R^2  DCV: 0.679 (+/-0.070)
43.67 seconds 

Top10
formula,P,Tc,AD
CaH3,200,191,1
ScH3,200,191,1
MnH3,200,191,1
H3Cl,200,191,1
KH3,200,191,1
TiH3,200,191,1
H3S,200,191,1
VH3,200,191,1
CrH3,200,191,1
CrH3,250,180,1


[1c3] SVR
Best parameters set found on development set:
{'model__C': 512.0, 'model__epsilon': 0.5, 'model__gamma': 4.0}
C:  RMSE, MAE, R^2 = 14.593, 7.097, 0.862
CV: RMSE, MAE, R^2 = 22.916, 14.316, 0.661
P:  RMSE, MAE, R^2 = 50.052, 48.450, 0.000

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 29.948 (+/-1.981)
MAE  DCV: 19.936 (+/-1.017)
R^2  DCV: 0.418 (+/-0.077)
8053.72 seconds

Top10
formula,P,Tc,AD
VH3,200,268,1
CrH3,200,266,1
TiH3,200,266,1
ScH3,200,261,1
MnH3,200,261,1
TiH3,250,258,1
VH3,250,258,1
CrH3,250,256,1
ScH3,250,254,1
CaH3,200,253,1

※探索範囲を小さくしたver.
Best parameters set found on development set:
{'model__C': 1024.0, 'model__epsilon': 1.0, 'model__gamma': 32.0}
C:  RMSE, MAE, R^2 =  6.327,  3.921, 0.974
CV: RMSE, MAE, R^2 = 21.247, 13.764, 0.708
P:  RMSE, MAE, R^2 = 43.557, 43.197, 0.000

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 29.030 (+/-2.159)
MAE  DCV: 19.685 (+/-1.321)
R^2  DCV: 0.452 (+/-0.082)
373.08 seconds 

Top10
formula,P,Tc,AD
KH3,200,299,1
H3Cl,200,271,1
CaH3,200,255,1
H3Cl,150,249,1
KH3,150,242,1
H3Cl,250,218,1
KH3,250,211,1
H3S,150,205,1
CaH3,150,202,1
TiH6,150,197,1

ref:
過去の計算事例
07/27 k-NN
08/09 RF
