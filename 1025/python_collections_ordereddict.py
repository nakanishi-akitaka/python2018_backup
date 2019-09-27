# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-collections-ordereddict/
Created on Thu Oct 25 12:14:15 2018

@author: Akitaka
"""
# import collections
from collections import OrderedDict
od = OrderedDict()
od['k1'] = 1
od['k2'] = 2
od['k3'] = 3
print(od)

print(OrderedDict(k1=1, k2=2, k3=3))
print(OrderedDict([('k1', 1), ('k2', 2), ('k3', 3)]))
print(OrderedDict((['k1', 1], ['k2', 2], ['k3', 3])))
print(OrderedDict({'k1': 1, 'k2': 2, 'k3': 3}))

#%%
print(issubclass(OrderedDict,dict))
print(od['k1'])

od['k2'] = 200
print(od)

od.update(k4=4, k5=5)
print(od)

del od['k4'], od['k5']
print(od)

#%%
od.move_to_end('k1')
print(od)

od.move_to_end('k1', False)
print(od)

#%%
l = list(od.items())
print(l)

l.insert(1, ('kx', -1))
print(l)

od = OrderedDict(l)
print(od)

#%%
l = list(od.items())
print(l)

l[0], l[2] = l[2], l[0]
print(l)

od = OrderedDict(l)
print(od)

#%%
l = list(od.items())
k = list(od.keys())
print(k)
print(k.index('kx'))

l[k.index('kx')], l[k.index('k3')] = l[k.index('k3')], l[k.index('kx')]
print(l)

od = OrderedDict(l)
print(od)

#%%
print(od)
od_sorted_key = OrderedDict(
        sorted(od.items(), key=lambda x: x[0]))
print(od_sorted_key)

od_sorted_value = OrderedDict(
        sorted(od.items(), key=lambda x: x[1], reverse=True))
print(od_sorted_value)

