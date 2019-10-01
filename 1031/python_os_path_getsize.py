# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-os-path-getsize/
Created on Wed Oct 31 11:31:10 2018

@author: Akitaka
"""
import os

print(os.path.getsize('test.txt'))

#%%
def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

print(get_dir_size('new-dir'))

#%%
def get_dir_size_old(path='.'):
    total = 0
    for p in os.listdir(path):
        full_path = os.path.join(path, p)
        if os.path.isfile(full_path):
            total += os.path.getsize(full_path)
        elif os.path.isdir(full_path):
            total += get_dir_size_old(full_path)
    return total

print(get_dir_size_old('new-dir'))

