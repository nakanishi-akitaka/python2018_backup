# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-get-key-from-value/
Created on Thu Oct 25 11:20:46 2018

@author: Akitaka
"""
d = {'key1': 'aaa', 'key2': 'aaa', 'key3': 'bbb'}
value = d['key1']
print(value)

#%%
keys = [k for k, v in d.items() if v == 'aaa']
print(keys)
keys = [k for k, v in d.items() if v == 'bbb']
print(keys)
keys = [k for k, v in d.items() if v == 'xxx']
print(keys)

#%%
keys = [k for k, v in d.items() if v == 'aaa'][0]
print(keys)
keys = [k for k, v in d.items() if v == 'bbb'][0]
print(keys)

#%%
def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

keys = get_keys_from_value(d, 'aaa')
print(keys)


#%%
def get_keys_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None

key = get_keys_from_value(d, 'aaa')
print(key)
key = get_keys_from_value(d, 'bbb')
print(key)
key = get_keys_from_value(d, 'xxx')
print(key)

#%%
d_num = {'key1': 1, 'key2': 2, 'key3': 3}
keys = [k for k, v in d_num.items() if v >= 2]
print(keys)

keys = [k for k, v in d_num.items() if v % 2 == 1]
print(keys)

d_str = {'key1': 'aaa@xxx.com', 'key2': 'bbb@yyy.net', 'key3': 'ccc@zzz.com'}
keys = [k for k, v in d_str.items() if v.endswith('com')]
print(keys)
