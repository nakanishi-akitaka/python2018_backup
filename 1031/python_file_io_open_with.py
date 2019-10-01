# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-file-io-open-with/
Created on Wed Oct 31 10:38:12 2018

@author: Akitaka
"""
path = 'test.txt'

f = open(path)

print(type(f))
# <class '_io.TextIOWrapper'>

f.close()

with open(path) as f:
    print(type(f))
# <class '_io.TextIOWrapper'>
    
#%%
with open(path) as f:
    s = f.read()
    print(type(s))
    print(s)
# <class 'str'>
# line 1
# line 2
# line 3

#%%
with open(path) as f:
    s = f.read()

print(s)
# line 1
# line 2
# line 3

#%%
with open(path) as f:
    l = f.readlines()
    print(type(l))
    print(l)
# <class 'list'>
# ['line 1\n', 'line 2\n', 'line 3']

#%%
with open(path) as f:
    l_strip = [s.strip() for s in f.readlines()]
    print(l_strip)
# ['line 1', 'line 2', 'line 3']

#%%
with open(path) as f:
    l = f.readlines()
    print(l[1])
# line 2
#

#%%
with open(path) as f:
    for s_line in f:
        print(s_line)
# line 1
#
# line 2
#
# line 3
#

#%%
with open(path) as f:
    s_line = f.readline()
    print(s_line)
# line 1
#

#%%
with open(path) as f:
    while True:
        s_line = f.readline()
        print(s_line)
        if not s_line:
            break
# line 1
#
# line 2
#
# line 3
#
#%%
path_w = 'test_w.txt'

s = 'New file'

with open(path_w, mode='w') as f:
    f.write(s)

with open(path_w) as f:
    print(f.read())
# New file

#%%
s = 'New file'

with open(path_w, mode='w') as f:
    f.write(s)

with open(path_w) as f:
    print(f.read())
# New file

#%%
s = 'New line 1\nNew line 2\nNew line 3'

with open(path_w, mode='w') as f:
    f.write(s)

with open(path_w) as f:
    print(f.read())
# New line 1
# New line 2
# New line 3

#%%
l = ['One', 'Two', 'Three']

with open(path_w, mode='w') as f:
    f.writelines(l)

with open(path_w) as f:
    print(f.read())
# OneTwoThree

#%%
with open(path_w, mode='w') as f:
    f.write('\n'.join(l))

with open(path_w) as f:
    print(f.read())
# One
# Two
# Three

#%%
with open('empty.txt', 'w'):
    pass


#%%
try:
    with open(path_w, mode='x') as f:
        f.write(s)
except FileExistsError:
    pass

#%%
import os

if not os.path.isfile(path_w):
    with open(path_w, mode='w') as f:
        f.write(s)

#%%
with open(path_w, mode='a') as f:
    f.write('Four')

with open(path_w) as f:
    print(f.read())
# One
# Two
# ThreeFour

#%%
with open(path_w, mode='a') as f:
    f.write('\nFour')

with open(path_w) as f:
    print(f.read())
# One
# Two
# ThreeFour
# Four

#%%
with open(path_w, mode='r+') as f:
    f.write('123456')

with open(path_w) as f:
    print(f.read())
# 123456o
# ThreeFour
# Four

#%%
with open(path_w, mode='r+') as f:
    f.seek(3)
    f.write('-')

with open(path_w) as f:
    print(f.read())
# 123-56o
# ThreeFour
# Four

#%%
with open(path_w) as f:
    l = f.readlines()

l.insert(0, 'FIRST\n')

with open(path_w, mode='w') as f:
    f.writelines(l)

with open(path_w) as f:
    print(f.read())
# FIRST
# 123-56o
# ThreeFour
# Four

















