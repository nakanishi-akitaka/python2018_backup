# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-os-basename-dirname-split-splitext/
Created on Wed Oct 31 11:35:10 2018

@author: Akitaka
"""

import os

filepath = './new-dir/test.txt'
print(os.sep)


print(os.sep is os.path.sep)


#%%
basename = os.path.basename(filepath)

print(basename)

print(type(basename))

#%%
dirname = os.path.dirname(filepath)

print(dirname)

print(type(dirname))

#%%
base_dir_pair = os.path.split(filepath)

print(base_dir_pair)

print(type(base_dir_pair))

print(os.path.split(filepath)[0] == os.path.dirname(filepath))
print(os.path.split(filepath)[1] == os.path.basename(filepath))

dirname, basename = os.path.split(filepath)

print(dirname)
print(basename)

#%%
dirpath_without_sep = './new-dir/new-sub-dir'

print(os.path.split(dirpath_without_sep))

print(os.path.basename(dirpath_without_sep))

dirpath_without_sep = './new-dir/new-sub-dir/'

print(os.path.split(dirpath_without_sep))

print(os.path.basename(dirpath_without_sep))

#%%
root_ext_pair = os.path.splitext(filepath)

print(root_ext_pair)

print(type(root_ext_pair))

root, ext = os.path.splitext(filepath)

path = root + ext

print(path)

#%%
other_ext_filepath = os.path.splitext(filepath)[0] + '.jpg'

print(other_ext_filepath)

#%%
ext_without_period = os.path.splitext(filepath)[1][1:]

print(ext_without_period)

#%%
path = os.path.join('dir', 'subdir', 'filename.ext')
print(path)

#%%
other_filepath = os.path.join(os.path.dirname(filepath), 'other_file.ext')

print(other_filepath)

#%%
import ntpath

print(ntpath.sep)
# \

print('\\')
# \

print(ntpath.sep is '\\')
# True

#%%
file_path = 'c:\\dir\\subdir\\filename.ext'
file_path_raw = r'c:\dir\subdir\filename.ext'

print(file_path == file_path_raw)
# True

#%%
print(ntpath.basename(file_path))

print(ntpath.dirname(file_path))

print(ntpath.split(file_path))

#%%
print(ntpath.splitdrive(file_path))

drive_letter = ntpath.splitdrive(file_path)[0][0]
print(type(ntpath.splitdrive(file_path)))
print(type(ntpath.splitdrive(file_path)[0]))
print(drive_letter)

print(ntpath.join('c:', 'dir', 'subdir', 'filename.ext'))
print(ntpath.join('c:', ntpath.sep, 'dir', 'subdir', 'filename.ext'))
print(ntpath.join('c:\\', 'dir', 'subdir', 'filename.ext'))




















