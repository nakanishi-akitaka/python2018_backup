# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-comment/
Created on Thu Oct 18 13:46:11 2018

@author: Akitaka
"""

a = 1  # comment
a = 1  # comment # b = 2
# a = 1

a = 1
# b = 2
# c = 3
# d = 4
e = 5

"""
E261: at least two spaces before inline comment
"""

# No:
a = 1 # comment

# Yes:
a = 1  # comment

a = 1      # comment
xyz = 100  # comment

"""
E262: inline comment should start with '# '
"""
# No:
a = 1  #comment
a = 1  #  comment

# Yes:
a = 1  # comment

"""
E265: block comment should start with '# '
"""

# No:
#comment

# Yes:
# comment
#     indented comment

"""
E266: too many leading '#' for block comment
"""

# No:
## comment

# Yes:
# comment

def test(a, b):
    '''docstring
    description
    '''
    print(a)
    print(b)

a = 1
'''
b = 2
c = 3
d = 4
'''
e = 5

def test2(a, b):
    print(a)
    '''
    comment line1
    comment line2
    comment line3
    '''
    print(b)

#def test3(a, b):
#    print(a)
#'''
#comment line1
#comment line2
#comment line3
#'''
#    print(b)

def test4(a, b):
    print(a)
    # comment line1
    # comment line2
    # comment line3
    print(b)

def test5(a, b):
    print(a)
# comment line1
# comment line2
# comment line3
    print(b)

def func_annotations_type(x: str, y: int) -> str:
    return x * y

print(func_annotations_type('abc', 3))

