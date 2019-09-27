# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-list-values/
Created on Thu Oct 25 12:02:02 2018

@author: Akitaka
"""
l = [{'Name': 'Alice', 'Age': 40, 'Point': 80}, 
     {'Name': 'Bob', 'Age': 20},
     {'Name': 'Charlie', 'Age': 30, 'Point': 70}]

l_name = [d.get('Name') for d in l]
print(l_name)

l_age= [d.get('Age') for d in l]
print(l_age)

l_point = [d.get('Point') for d in l]
print(l_point)

#%%
l_name = [d['Name'] for d in l]
print(l_name)

l_point = [d.get('Point', 0) for d in l]
print(l_point)

l_point_ignore = [d.get('Point') for d in l if d.get('Point')]
print(l_point_ignore)




