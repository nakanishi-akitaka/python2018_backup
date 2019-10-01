# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-os-exists-isfile-isdir/
Created on Wed Oct 31 11:33:37 2018

@author: Akitaka
"""

import os

filepath = 'test.txt'
dirpath = 'new-dir'

print(os.path.exists(filepath))
# True

print(os.path.exists(dirpath))
# True

#%%
print(os.path.isfile(filepath))
# True

print(os.path.isfile(dirpath))
# False

#%%
print(os.path.isdir(filepath))
# False

print(os.path.isdir(dirpath))
# True
