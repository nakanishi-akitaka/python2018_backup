# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-str-num-conversion/
Created on Wed Oct 24 10:20:51 2018

@author: Akitaka
"""

l_n = [-0.5, 0, 1.0, 100, 1.2e-2, 0xff, 0b11]
l_n_str = [str(n) for n in l_n]
print(l_n_str)

#%%
l_i = [0, 64, 128, 192, 256]
l_i_hex1 = [hex(i) for i in l_i]
print(l_i_hex1)

l_i_hex2 = [format(i, '04x') for i in l_i]
print(l_i_hex2)

l_i_hex3 = [format(i, '#06x') for i in l_i]
print(l_i_hex3)

#%%
l_f = [0.0001, 123.456, 123400000]
l_f_e1 = [format(f, 'e') for f in l_f]
print(l_f_e1)

l_f_e2 = [format(f, '.3E') for f in l_f]
print(l_f_e2)

#%%
l_si = ['-10', '0', '100']
l_si_i = [int(s) for s in l_si]
print(l_si_i)

l_sf = ['.123', '1.23', '123']
l_sf_f = [float(s) for s in l_sf]
print(l_sf_f)

#%%
l_sb = ['0011', '0101', '1111']
l_sb_i = [int(s, 2) for s in l_sb]
print(l_sb_i)

l_sbox = ['100', '0b100', '0o77', '0xff']
l_sbox_i = [int(s, 0) for s in l_sbox]
print(l_sbox_i)

#%%
l_se = ['1.23e3', '0.123e-1', '123']
l_se_f = [float(s) for s in l_se]
print(l_se_f)

#%%
def is_int(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True

def is_float(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

l_multi = ['-100', '100', '1.23', '1.23e2', 'one']
l_multi_i = [int(s) for s in l_multi if is_int(s)]
print(l_multi_i)
l_multi_f = [float(s) for s in l_multi if is_float(s)]
print(l_multi_f)



