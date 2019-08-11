
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

digits = datasets.load_digits()

print(digits.data.shape)
# (1797, 64)

print(digits.target.shape)
# (1797,)

X_reduced = PCA(n_components=2).fit_transform(digits.data)

print(X_reduced.shape)

# (1797, 2)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=digits.target)
plt.colorbar()


# # 使い方メモ
# 1. `Code`の部分をクリックしてMarkdownを選択
# 2. Markdownを書き込みたいセルにMarkdownで記入
# 3. `Ctrl`+`Enter`で記入したセルを実行
# 4. Markdownが変換される
# 5. 編集したい場合は、セルをダブルクリック
# 
# # 見出し１
# ## 見出し２
# ### 箇条書きの例
# -----------------
# * 箇条書き１
# * 箇条書き２
# ------
# コード`code`を等幅で記載
# ```
#     print('Hello world')
# ```
# ### リンク
# https://www.google.com  
# [Google](https://www.google.com)
# 
# 
