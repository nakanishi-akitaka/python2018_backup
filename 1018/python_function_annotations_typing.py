# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-function-annotations-typing/
Created on Thu Oct 18 14:12:50 2018

@author: Akitaka
"""

def func(x, y):
    return x * y

print(func('abc', 3))
print(func(4, 3))

def func_annotations(x: 'description-x', y: 'description-y') -> 'description-return':
    return x * y

print(func_annotations('abc', 3))
print(func_annotations(4, 3))

def func_annotations_default(x: 'description-x', y: 'description-y' = 3) -> 'description-return':
    return x * y

print(func_annotations_default('abc'))
print(func_annotations_default(4))

def func_annotations_type(x: str, y:  int) -> str:
    return x * y

print(func_annotations_type('abc', 3))
print(func_annotations_type(4, 3))

print(type(func_annotations.__annotations__))
print(func_annotations.__annotations__)
print(func_annotations.__annotations__['x'])

from typing import Union, List
def func_u(x: List[Union[int, float]]) -> float:
    return sum(x) ** 0.5

print(func_u([0.5, 9.5, 90]))

