# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-divmod-quotient-remainder/
Created on Tue Oct 30 11:18:48 2018

@author: Akitaka
"""

q = 10 // 3
mod = 10 % 3
print(q, mod)
# 3 1

q, mod = divmod(10, 3)
print(q, mod)
# 3 1

answer = divmod(10, 3)
print(answer)
print(answer[0], answer[1])
# (3, 1)
# 3 1














