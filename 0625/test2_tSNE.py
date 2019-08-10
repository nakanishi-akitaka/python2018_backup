
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

digits = datasets.load_digits()

print(digits.data.shape)
# (1797, 64)

print(digits.target.shape)
# (1797,)

X_reduced = TSNE(n_components=2, random_state=0).fit_transform(digits.data)

print(X_reduced.shape)
# (1797, 2)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=digits.target)
plt.colorbar()
# <matplotlib.colorbar.Colorbar at 0x7ff21173ee90>

