
# coding: utf-8

# In[2]:


# https://pythondatascience.plavox.info/numpy
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
print(x)


# In[5]:


# https://pythondatascience.plavox.info/numpy/行列を作ってみよう

# 2 x 3 行列を作成
x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

# 配列自体の型を確認
print(type(x))

# 配列の要素の型を確認
print(x.dtype)

# 配列のサイズを確認
print(x.shape)


# In[12]:


# https://pythondatascience.plavox.info/numpy/行列を操作する

# 2 x 3 行列を作成
x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

# 特定の要素の値を取得
print(x[1, 2])
print(x[1][2])

# 0 行目のみを取得
print(x[0])

# 1 行目のみを取得
print(x[1])

# 1次元配列を作成
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 2番目の要素から7番目の要素までを取得
print(x[2:7])

# 後ろから2番目の要素から、前から10番目の要素までを取得
print(x[-2:10])

# 1番目の要素から7番目の要素まで１つスキップしながら取得
print(x[1:7:2])

# 1番目の要素から7番目の要素まで2つスキップしながら取得
print(x[1:7:3])

# 5番目の要素から最後の要素までを抽出
print(x[5:])


# In[32]:


# https://pythondatascience.plavox.info/numpy/数学系の関数
print(np.sin(0))
print(np.sin([0, 1]))
print(np.cos(1))
print(np.cos([0, 1]))
print(np.tan(1))
print(np.tan([0, 1]))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
x = np.linspace(-np.pi, np.pi, 201)
plt.plot(x, np.sin(x))
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()

print(np.arcsin(0))
print(np.arcsin([0, 1]))
print(np.arccos(1))
print(np.arccos([0, 1]))
print(np.arctan(1))
print(np.arctan([0, 1]))


# In[33]:


# https://pythondatascience.plavox.info/numpy/数学系の関数

# 度からラジアンへ
print(np.radians(0))
print(np.radians([np.rad2deg(1), 0, 90]))

# 度からラジアンへ
print(np.deg2rad(0))
print(np.deg2rad([np.rad2deg(1), 0, 90]))

# ラジアンから度へ
print(np.rad2deg(0))
print(np.rad2deg([np.deg2rad(180), 1, 0.5]))


# In[34]:


# https://pythondatascience.plavox.info/numpy/数学系の関数

# e(0)
print(np.exp(0)) 
# e(1)
print(np.exp(1))

# e を出力
print(np.e) 

# eが底の指数関数 (Log_e(x))
print(np.log(1))
#print(np.log([1, np.e, np.e ** 2, 0]))
print(np.log([1, np.e, np.e ** 2]))
 
# 2が底の指数関数 (Log_2(x))
print(np.log2(2))
# print(np.log2([1, np.e, np.e ** 2, 0]))
print(np.log2([1, np.e, np.e ** 2]))
 
# 10 が底の指数関数 (Log_10(x))
print(np.log10(1000))
print(np.log10(1e+10))  # 1e+10 = 10,000,000,000


# In[35]:


# https://pythondatascience.plavox.info/numpy/数学系の関数

# 配列 [1, 2, 3, 4, 5, 6] の各要素に対する 3 の剰余
print(np.mod([1, 2, 3, 4, 5, 6], 3))

# 配列 [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] を作成
a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
 
# 四捨五入 (小数点以下 .5 以上は繰上げ、.5未満は切捨て)
print(np.round(a))
 
# 切り捨て (小数部分を取り除く)
print(np.trunc(a))
 
# 切り捨て (小さい側の整数に丸める)
print(np.floor(a))
 
# 切り上げ (大きい側の整数に丸める)
print(np.ceil(a))
 
# ゼロに近い側の整数に丸める
print(np.fix(a))


# In[36]:


# https://pythondatascience.plavox.info/numpy/数学系の関数

# 配列 a, b を定義
a = [1, 4, 5, np.nan]
b = [2, 3, 6, 7]
 
# 各要素の最大値、最小値
print(np.maximum(a, b))
print(np.minimum(a, b))
 
# 各要素の最大値、最小値（NaN を無視する）
print(np.fmin(a, b))
print(np.fmax(a, b))
 
# なお、1つの配列の最大値、最小値は NumPy を使わずにも計算可能です。
print(max([2, 6, 1, 4, 5, 3]))
print(min([2, 6, 1, 4, 5, 3]))


# In[37]:


# https://pythondatascience.plavox.info/numpy/数学系の関数

# √16
print(np.sqrt(16))
 
# 配列を引数に取ることも可能です。例: [√1, √4, √9]
print(np.sqrt([1, 4, 9]))
 
# 配列でなければ、NumPy を用いずに計算できます。
import math
print(math.sqrt(16))


# In[41]:


# https://pythondatascience.plavox.info/numpy/数学系の関数

# 1+2j の実数部 (=1) を返す
print(np.real(1 + 2j))

# [2+3j, 2-3j] の実数部
print(np.real([2+3j, 2-3j]))

# 1+2j の虚数部 (=2) を返す
print(np.imag(1 + 2j))

# [2+3j, 2-3j] の虚数部
print(np.imag([2+3j, 2-3j]))
 
# 1+2j の共役複素数
print(np.conj(1+2j))

# [2+3j, 2-3j] の共役複素数
print(np.conj([2+3j, 2-3j]))

# |-2|
print(np.absolute(-2))
 
# |[-1.2, 1.2]|
print(np.absolute([-1.2, 1.2]))

