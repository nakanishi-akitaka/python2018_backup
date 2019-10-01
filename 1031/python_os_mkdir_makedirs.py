# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-os-mkdir-makedirs/
Created on Wed Oct 31 11:28:21 2018

@author: Akitaka
"""

import os

new_dir_path = 'new-dir'

os.mkdir(new_dir_path)

#%%
new_dir_path_recursive = 'new-dir/new-sub-dir'

os.makedirs(new_dir_path_recursive)

#%%
os.makedirs(new_dir_path_recursive, exist_ok=True)

try:
    os.makedirs(new_dir_path_recursive)
except FileExistsError:
    pass

def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

my_makedirs(new_dir_path_recursive)