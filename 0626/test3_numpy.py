
# coding: utf-8

# In[2]:


# https://pythondatascience.plavox.info/numpy/線形代数

import numpy as np
 
# 整数同士のドット積（通常の掛け算と同じになります）
# 3・4
print(np.dot(3, 4))
 
# 行列同士のドット積
# (2, 3)・(2, 3)
print(np.dot([2, 3], [2, 3])) 

# 複素数を扱うこともできます
# (2j, 3j)・(2j, 3j)
print(np.dot([2j, 3j], [2j, 3j]))
 
# 2 次元行列同士のドット積を計算できます。
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
print(np.dot(a, b))


# In[4]:


# 通常のベクトル同士の内積
# (1, 2, 3)・(0, 1, 0)
a = np.array([1, 2, 3])
b = np.array([0, 1, 0])
print(np.inner(a, b))
 
# 多次元ベクトルの内積を計算
a = np.arange(24).reshape((2,3,4))
print(a)
b = np.arange(4)
print(b)
print(np.inner(a, b))
 
# b がスカラーの場合は以下のように指定します
a = np.eye(2)
print(a)
print(np.inner(np.eye(2), 7))


# In[6]:


##マンデルブロット (Mandelbrot)
rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
print(rl)
 
im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
print(im)
 
grid = rl + im
print(grid)
 
# 文字列をベクトルとして計算することもできます
x = np.array(['a', 'b', 'c'], dtype=object)
print(np.outer(x, [1, 2, 3]))


# In[7]:


# 3 次の単位行列
print(np.identity(3))
 
# 4 次の単位行列
print(np.identity(4))


# In[10]:


# numpy.linalg パッケージを読み込み
from numpy import linalg as LA
 
# 2 次行列 [[3, 1], [2, 4]] の固有値、固有ベクトル
matrix1 = np.array([[3, 1], [2, 4]])
w, v = LA.eig(matrix1)
print(w)  # 固有値
print(v)  # 固有ベクトル(正規化済)
 
# 3 次行列 [[1, -1, 2], [0, 2, -1], [0, 0, 3]] の固有値、固有ベクトル
matrix2 = np.array([[1, -1, 2], [0, 2, -1], [0, 0, 3]])
w, v = LA.eig(matrix2)
print(w)  # 固有値
print(v)  # 固有ベクトル(正規化済)

# 2 次行列 [[3, 1], [2, 4]] の固有値
print(LA.eigvals(matrix1))
 
# 3 次行列 [[1, -1, 2], [0, 2, -1], [0, 0, 3]] の固有値
print(LA.eigvals(matrix2))


# In[12]:


# 2次の正方行列 [[1, 2], [3, 4]] の行列式の値
a = np.array([[1, 2], [3, 4]])
print(np.linalg.det(a))

# 2 次の行列 A = [[1, 2], [3, 4]] の逆行列 A^-1
from numpy.linalg import inv
 
a = np.array([[1, 2], [3, 4]])
print(inv(a))
 
# 確認 (A・A^-1 = 単位行列 になることを確認)
ainv = inv(a)
print(np.dot(a, ainv))


# In[14]:


# numpy.linalg から matrix_rank 関数をロード
from numpy.linalg import matrix_rank
 
# 行列 [[1, 2, 3],[3, 2, 1],[2, 4, 6]] のランクを求めます。
matrix1 = np.array([[1, 2, 3], [3, 2, 1], [2, 4, 6]])
print(matrix1)
print(matrix_rank(matrix1))
 
# 対角行列の場合は常に次数と同じになります
matrix2 = np.eye(4)
print(matrix2)
print(matrix_rank(matrix2))


# In[15]:


# 長さが 3 の零行列
print(np.zeros(3))
 
# 長さが 4 x 4 の零行列
print(np.zeros([4, 4]))

from numpy import linalg as LA
 
a = np.array([1, 2, 3, 4])
print(LA.norm(a))
 
# 2 次の行列 b = [[1, 2], [3, 4]] のノルム ||b|| は以下のように計算できます。
b = np.array([[1, 2], [3, 4]])
print(LA.norm(b))

