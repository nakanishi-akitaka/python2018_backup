# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-doctest-example/
Created on Thu Oct 18 14:08:08 2018

@author: Akitaka
"""

def add(a, b):
    '''
    >>> add(1, 2)
    3
    >>> add(5, 10)
    15
    '''
    return a + b

if __name__ == '__main__':
    import doctest
    doctest.testmod()
#    doctest.testmod(verbose=True)

#    import sys
#    result = add(int(sys.argv[1]), int(sys.argv[2]))
#    print(result)
