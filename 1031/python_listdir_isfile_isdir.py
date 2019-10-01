# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-listdir-isfile-isdir/
Created on Wed Oct 31 11:45:21 2018

@author: Akitaka
"""
import os

path = "./testdir"

files = os.listdir(path)
print(type(files))  # <class 'list'>
print(files)        # ['dir1', 'dir2', 'file1', 'file2.txt', 'file3.jpg']

#%%
files = os.listdir(path)
files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
print(files_file)   # ['file1', 'file2.txt', 'file3.jpg']

#%%
files = os.listdir(path)
files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
print(files_dir)    # ['dir1', 'dir2']












