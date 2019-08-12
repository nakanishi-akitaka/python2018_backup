# -*- coding: utf-8 -*-
"""
ROC曲線とAUCについて定義と関係性をまとめたよ
https://qiita.com/koyamauchi/items/a2ed9f638b51f3b22cd6
Created on Mon Jul  2 13:02:41 2018

@author: Akitaka
"""

import numpy as np
from sklearn.metrics import roc_auc_score
# y = np.array([1, 1, 2, 2])
y = np.array([0, 0, 1, 1])
pred = np.array([0.1, 0.4, 0.35, 0.8])
print(roc_auc_score(y, pred))

