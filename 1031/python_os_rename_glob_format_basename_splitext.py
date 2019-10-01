# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-os-rename-glob-format-basename-splitext/
Created on Wed Oct 31 11:47:42 2018

@author: Akitaka
"""

import glob

print(glob.glob('./testdir/*'))

import os
os.rename('./testdir/a.jpg', './testdir/a_000.jpg')

# 3を2桁でゼロ埋め
print('{0:02d}'.format(3))
# => 03

# 4と6をそれぞれ3桁と4桁でゼロ埋め
print('{0:03d}, {1:04d}'.format(4, 6))
# => 004, 0006

import os
import glob

path = "./testdir"
files = glob.glob(path + '/*')

for f in files:
    os.rename(f, os.path.join(path, 'img_' + os.path.basename(f)))

for i, f in enumerate(files):
    os.rename(f, os.path.join(path, '{0:03d}'.format(i) +
                              '_' + os.path.basename(f)))

for i, f in enumerate(files, 1):
    os.rename(f, os.path.join(path, '{0:03d}'.format(i) +
                              '_' + os.path.basename(f)))

import os
import glob

files = glob.glob('./testdir/*')

for f in files:
    ftitle, fext = os.path.splitext(f)
    os.rename(f, ftitle + '_img' + fext)

for i, f in enumerate(files):
    ftitle, fext = os.path.splitext(f)
    os.rename(f, ftitle + '_' + '{0:03d}'.format(i) + fext)
    
    


