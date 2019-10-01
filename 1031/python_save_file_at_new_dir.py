# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-save-file-at-new-dir/
Created on Wed Oct 31 11:01:20 2018

@author: Akitaka
"""
open('not_exist_dir/new_file.txt', 'w')
# FileNotFoundError

import os

os.makedirs(new_dir_path, exist_ok=True)

if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)

#%%

def save_file_at_new_dir(new_dir_path, new_filename, new_file_content, mode='w'):
    os.makedirs(new_dir_path, exist_ok=True)
    with open(os.path.join(new_dir_path, new_filename), mode) as f:
        f.write(new_file_content)




