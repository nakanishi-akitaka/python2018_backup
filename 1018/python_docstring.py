# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-docstring/
Created on Thu Oct 18 13:55:23 2018

@author: Akitaka
"""

def my_func():
    """docstring-test
    line1
    line2
    line3
    """

print(my_func.__doc__)
print(type(my_func.__doc__))

help(my_func)

def my_func2():
    'docstring-test'

print(my_func2.__doc__)

def my_func_error():
    a = 100
    """docstring-test
    line1
    line2
    line3
    """

print(my_func_error.__doc__)

class MyClass:
    """docstring-test
    line1
    line2
    line3
    """

print(MyClass.__doc__)

def add(a, b):
    '''
    >>> add(1, 2)
    3
    >>> add(5, 19)
    15
    '''
    return a + b

def func_rest(arg1, arg2):
    """Summary line.
    
    :param arg1: Description of arg1
    :type arg1: int
    :param arg2: Description of arg2
    :type arg2: str
    :returns: Description of return value
    :rtype: bool
    """
    return True

def func_numpy(arg1, arg2):
    """Summary line.
    
    Extended description of function.
    
    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value

    """
    return True

def func_goole(arg1, arg2):
    """summary line.
    
    Extended description of function.
    
    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2
    Returns:
        bool: Description of return value
    """
    return True




