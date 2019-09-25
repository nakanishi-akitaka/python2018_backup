# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-common/
Created on Tue Oct 23 15:10:32 2018

@author: Akitaka
"""

l1 = ['a', 'b', 'c']
l2 = ['b', 'c', 'd']
l3 = ['c', 'd', 'e']

l1_l2_and = set(l1) & set(l2)
print(l1_l2_and)
print(type(l1_l2_and))

l1_l2_and_list = list(set(l1) & set(l2))
print(l1_l2_and_list)
print(type(l1_l2_and_list))

print(len(l1_l2_and))

l1_l2_l3_and = set(l1) & set(l2) & set(l3)
print(l1_l2_l3_and)

#%%
l1_l2_sym_diff = set(l1) ^ set(l2)
print(l1_l2_sym_diff)
print(list(l1_l2_sym_diff))
print(len(l1_l2_sym_diff))

l1_l2_l3_sym_diff = set(l1) ^ set(l2) ^ set(l3)
print(l1_l2_l3_sym_diff)

#%%
l_all = l1 + l2 + l3
print(l_all)

l_all_only = [x for x in l_all if l_all.count(x) == 1]
print(l_all_only)

l1_duplicate = ['a', 'a', 'b', 'c']
l_duplicate_all = l1_duplicate + l2 + l3
l_duplicate_all_only = [x for x in l_duplicate_all if l_duplicate_all.count(x) == 1]
print(l_duplicate_all_only)

#%%
l_unique_all = list(set(l1_duplicate)) + list(set(l2)) + list(set(l3))
print(l_unique_all)

l_uniques_all_only = [x for x in l_unique_all if l_unique_all.count(x) == 1]
print(l_uniques_all_only)

#%%
l1_l2_or = set(l1 + l2)
print(l1_l2_or)
print(list(l1_l2_or))
print(len(l1_l2_or))

l1_l2_l3_or = set(l1 + l2 + l3)
print(l1_l2_l3_or)

