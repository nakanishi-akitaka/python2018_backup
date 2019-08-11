
# coding: utf-8

# In[4]:


# https://pythondatascience.plavox.info/numpy/乱数を生成
# 一様乱数 (0.0 – 1.0) の間のランダムな数値を出力するには、
# numpy.random.rand(出力する件数) を用います。

import numpy as np

# 一様乱数 を 1 件出力する
print(np.random.rand())

# 一様乱数 を 3 件出力する
print(np.random.rand(3))

# 一様乱数 を 3  x 4 の行列で出力する
print(np.random.rand(3, 4))


# In[7]:


# 正規分布に従う乱数を出力するには、numpy.random.normal(平均, 分散, 出力する件数)を用います。
# 引数を省略した場合、平均=0.0, 分散=1.0, 出力する件数= 1 件 で出力されます。
# 標準正規乱数 (平均:0.0, 分散:1.0) に従う乱数を 1 件出力
print(np.random.normal())
 
# 平均:50, 分散:10 の正規分布に従う乱数を 1 件出力
print(np.random.normal(50, 10))
 
# 平均:50, 分散:10 の正規分布に従う乱数を 10 件出力
print(np.random.normal(50, 10, 10))

# 平均:50, 分散:10 の正規分布に従う乱数を 3  x 4 の行列で出力する
print(np.random.normal(50, 10, (3,4)))


# In[8]:


# 特定の区間の乱数を出力するには、
# numpy.random.randint(下限[, 上限,[, 出力する件数]]) を用います。

# 0-10 の間の整数を 1 件出力する
print(np.random.randint(10))
 
# 10-15 の間の整数を 1 件出力する
print(np.random.randint(10, 15))
 
# 1-100 の間の整数を 10 件出力する
print(np.random.randint(1, 100, 10))


# In[9]:


# 配列の順番をランダムに並び替えるには、
# numpy.random.shuffle(シャッフル対象の配列) を用います。
arr1 = ['A', 'B', 'C', 'D', 'E']
np.random.shuffle(arr1)
print(arr1)


# In[11]:


# numpy.random.seed(seed=シードに用いる値) をシード (種) を指定することで、
# 発生する乱数をあらかじめ固定することが可能です。
# 乱数を用いる分析や処理で、再現性が必要な場合などに用いられます。

# シードを 32 に設定して乱数を出力
np.random.seed(seed=32)
print(np.random.rand())

# シードを 32 に設定して乱数を出力 (同じ乱数が出力されます)
np.random.seed(seed=32)
print(np.random.rand())

