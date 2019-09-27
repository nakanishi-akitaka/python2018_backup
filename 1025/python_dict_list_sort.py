# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-list-sort/
Created on Thu Oct 25 12:10:10 2018

@author: Akitaka
"""

import pprint

l = [{'Name': 'Alice', 'Age': 40, 'Point': 80}, 
     {'Name': 'Bob', 'Age': 20},
     {'Name': 'Charlie', 'Age': 30, 'Point': 70}]

l.sort(key=lambda x: x['Age'])
pprint.pprint(l)

l.sort(key=lambda x: x['Name'])
pprint.pprint(l)

l.sort(key=lambda x: x.get('Point', 0))
pprint.pprint(l)

l.sort(key=lambda x: x.get('Point', 100))
pprint.pprint(l)

#%%
l.sort(key=lambda x: x['Name'], reverse=True)
pprint.pprint(l)

l_sorted = sorted(l, key=lambda x: x['Age'], reverse=True)
pprint.pprint(l_sorted)

