# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-sort-sorted/
Created on Tue Oct 23 12:46:12 2018

@author: Akitaka
"""

org_list = [3, 1, 4, 5, 2]
org_list.sort()
print(org_list)

org_list.sort(reverse=True)
print(org_list)


#%%
org_list = [3, 1, 4, 5, 2]
new_list = sorted(org_list)
print(org_list)
print(new_list)

new_list_reverse = sorted(org_list, reverse=True)
print(org_list)
print(new_list_reverse)

#%%
org_str = 'cebad'
new_str_list = sorted(org_str)
print(org_str)
print(new_str_list)

new_str = ''.join(new_str_list)
print(new_str)

new_str = ''.join(sorted(org_str))
print(new_str)

new_str_reverse = ''.join(sorted(org_str, reverse=True))
print(new_str_reverse)

#%%
org_tuple = (3, 1, 4, 5,2)
new_tuple_list = sorted(org_tuple)
print(org_tuple)
print(new_tuple_list)

new_tuple = tuple(new_tuple_list)
print(new_tuple)

new_tuple = tuple(sorted(new_tuple_list))
print(new_tuple)

new_tuple_reverse = tuple(sorted(new_tuple_list, reverse=True))
print(new_tuple_reverse)
