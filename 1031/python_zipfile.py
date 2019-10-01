# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-zipfile/
Created on Wed Oct 31 11:51:33 2018

@author: Akitaka
"""

import zipfile

with zipfile.ZipFile('new_comp.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
    new_zip.write('test.txt')

#%%
with zipfile.ZipFile('new_comp.zip', 'a') as existing_zip:
    existing_zip.write('new-dir/test.txt', arcname='test2.txt')

#%%
with zipfile.ZipFile('new_comp.zip') as existing_zip:
    print(existing_zip.namelist())

#%%
with zipfile.ZipFile('new_comp.zip') as existing_zip:
    existing_zip.extractall('zip')

#%%
with zipfile.ZipFile('new_comp.zip') as existing_zip:
    existing_zip.extract('test.txt')



