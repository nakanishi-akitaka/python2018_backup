# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:04:13 2018

@author: Akitaka
"""

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import linear_model, metrics, preprocessing, cross_validation #機械学習用のライブラリを利用
from mlxtend.plotting import plot_decision_regions #学習結果をプロットする外部ライブラリを利用
from sklearn.kernel_approximation import RBFSampler #カーネル近似用の関数
from matplotlib.colors import ListedColormap #plot用
 
# 2：XORのデータを作成する(x=正、y=正)=0,(x=正、y=負)=1, 的な--------------
np.random.seed(0)
X_xor=np.random.randn(200,2) 
y_xor=np.logical_xor(X_xor[:,0]>0, X_xor[:,1]>0)
y_xor=np.where(y_xor,1,0)
pd.DataFrame(y_xor)  #この行を実行するとデータが見れる
 
# 3：プロットしてみる------------------------------------------------------
#%matplotlib inline
 
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==0, 0], X_xor[y_xor==0, 1], c='r', marker='s', label='0')
plt.legend(loc='best')
plt.show
 
 
# 4：データの整形-------------------------------------------------------
X_std=X_xor
z=y_xor
 
#解説 5：カーネル近似を適用する------------------------------------------
rbf_feature = RBFSampler(gamma=1, n_components=100, random_state=1)
 
X_std = rbf_feature.fit_transform(X_std)
print("X_stdの大きさ ",pd.DataFrame(X_std).shape)
#pd.DataFrame(X_std).to_clipboard() #これでクリップボードに保持できるのでエクセルに貼れる
 
# 6：機械学習で分類する---------------------------------------------------
clf_result=linear_model.SGDClassifier(loss="hinge") #loss="hinge", loss="log"
 
# 7：K分割交差検証（cross validation）で性能を評価する---------------------
scores=cross_validation.cross_val_score(clf_result, X_std, z, cv=10)
print("平均正解率 = ", scores.mean())
print("正解率の標準偏差 = ", scores.std())
 
# 8：トレーニングデータとテストデータに分けて実行してみる------------------
X_train, X_test, train_label, test_label=cross_validation.train_test_split(X_std,z, test_size=0.1, random_state=1)
clf_result.fit(X_train, train_label)
#正答率を求める
pre=clf_result.predict(X_test)
ac_score=metrics.accuracy_score(test_label,pre)
print("正答率 = ",ac_score)
 
# 解説 9：Plotする
x1_min, x1_max, x2_min, x2_max=-3, 3, -3, 3
resolution=0.02
xx1, xx2=np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
X=(np.array([xx1.ravel(), xx2.ravel()]).T)
plot_z=clf_result.predict(rbf_feature.fit_transform(X))
colors=('red','blue')
cmap=ListedColormap(colors[:len(np.unique(plot_z))])
plot_z=plot_z.reshape(xx1.shape)
plt.contourf(xx1,xx2, plot_z, alpha=0.4, cmap=cmap)