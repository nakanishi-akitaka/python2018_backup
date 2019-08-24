# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 11:02:26 2018

@author: Akitaka
"""


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
print(enc.n_values_)
print(enc.feature_indices_)
print(enc.transform([[0, 1, 1]]).toarray())


#    enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
#    
#    enc.n_values_  [2, 3, 4]
#    各説明変数のクラス数
#    
#    enc.feature_indices_ [0 2 5 9]
#    バイナリ変換したベクトルのどの成分で、元の各説明変数が表現されているか
#    0~2 = 0
#    2~5 = 1
#    5~9 = 2
#    
#    enc.transform([[0, 1, 1]]).toarray()
#    [0,1,1]を変換する
