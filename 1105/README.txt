# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:20:00 2018

@author: Akitaka
"""

[1b] 水素化物の超伝導予測
[1b1] ファイル統一
[todo] -> [done]
C:\Users\Akitaka\Downloads\python\1026\Tc_4model_AD_DCV.py
20181026:EN, RR, LASSO, kNN

C:\Users\Akitaka\Downloads\python\1105\Tc_6model_AD_DCV.py
20181105: ...+RF, SVR

Random Forest
Best parameters set found on development set:
{'model__max_features': 0.8}
C:  RMSE, MAE, R^2 = 8.236, 5.620, 0.956
CV: RMSE, MAE, R^2 = 19.062, 13.328, 0.765
P:  RMSE, MAE, R^2 = 46.904, 39.253, 0.000

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 21.343 (+/-1.253)
MAE  DCV: 13.945 (+/-0.650)
R^2  DCV: 0.705 (+/-0.035)

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 28.118 (+/-1.285)
MAE: 20.716 (+/-0.618)
R^2: 0.488 (+/-0.047)
111.23 seconds 


SVR
Best parameters set found on development set:
{'model__C': 1024.0, 'model__epsilon': 1.0, 'model__gamma': 1024.0}
C:  RMSE, MAE, R^2 = 5.854, 3.615, 0.978
CV: RMSE, MAE, R^2 = 25.132, 16.377, 0.592
P:  RMSE, MAE, R^2 = 45.290, 45.252, 0.000

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 29.284 (+/-1.447)
MAE  DCV: 19.486 (+/-0.684)
R^2  DCV: 0.444 (+/-0.054)

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 28.158 (+/-1.541)
MAE: 15.835 (+/-1.129)
R^2: 0.486 (+/-0.058)
61.22 seconds 


kNN             
Best parameters set found on development set:
{'model__n_neighbors': 2}
C:  RMSE, MAE, R^2 = 9.528, 5.602, 0.941
CV: RMSE, MAE, R^2 = 23.373, 14.106, 0.647
P:  RMSE, MAE, R^2 = 34.608, 21.794, 0.000

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 37.623 (+/-1.594)
MAE  DCV: 26.272 (+/-1.263)
R^2  DCV: 0.083 (+/-0.077)

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 37.263 (+/-0.240)
MAE: 29.070 (+/-0.389)
R^2: 0.102 (+/-0.012)
57.73 seconds 
