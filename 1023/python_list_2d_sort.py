# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-2d-sort/
Created on Tue Oct 23 13:00:10 2018

@author: Akitaka
"""
import pprint
print([100] > [-100])
print([1, 2, 100] > [1, 2, -100])
print([1, 2, 100] > [1, 100])

l_2d = [[2, 30, 100], [1, 20, 300], [3, 10, 200]]
pprint.pprint(l_2d, width=40)
l_2d.sort()
pprint.pprint(l_2d, width=40)

#%%
l_2d.sort(key=lambda x: x[1])
pprint.pprint(l_2d, width=40)

l_2d.sort(key=lambda x: x[2], reverse=True)
pprint.pprint(l_2d, width=40)

l_sorted = sorted(l_2d, key=lambda x: x[0], reverse=True)
pprint.pprint(l_sorted, width=40)

#%%
l_3d = [[[0, 1, 2], [2, 30, 100]], [[3, 4, 5], [1, 20, 300]], [[6, 7, 8], [3, 10, 200]]]
pprint.pprint(l_3d, width=40)

l_sorted = sorted(l_3d, key=lambda x: x[1][0])
pprint.pprint(l_sorted, width=40)
