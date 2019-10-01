# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-makedirs-exist-ok/
Created on Wed Oct 31 11:29:54 2018

@author: Akitaka
"""

import os

# os.mkdir('not_exist_dir/new_dir')
# FileNotFoundError
os.makedirs('not_exist_dir/new_dir')

