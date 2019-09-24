# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-str-select-replace/
Created on Mon Oct 22 16:19:21 2018

@author: Akitaka
"""
l = ['oneXXXaaa', 'twoXXXbbb', 'three999aaa', '000111222']
l_in = [s for s in l if 'XXX' in s]
print(l_in)
l_in_not = [s for s in l if 'XXX' not in s]
print(l_in_not)

l_replace = [s.replace('XXX', 'ZZZ') for s in l]
print(l_replace)

l_replace_all = ['ZZZ' if 'XXX' in s else s for s in l]
print(l_replace_all)

#%%
l_start = [s for s in l if s.startswith('t')]
print(l_start)
l_start_not = [s for s in l if not s.startswith('t')]
print(l_start_not)

l_end = [s for s in l if s.endswith('aaa')]
print(l_end)
l_end_not = [s for s in l if not s.endswith('aaa')]
print(l_end_not)

#%%
l_lower = [s for s in l if s.islower()]
print(l_lower)

l_upper_all = [s.upper() for s in l]
print(l_upper_all)

l_lower_to_upper = [s.upper() if s.islower() else s for s in l]
print(l_lower_to_upper)

#%%
l_isalpha = [s for s in l if s.isalpha()]
print(l_isalpha)

l_isnumeric = [s for s in l if s.isnumeric()]
print(l_isnumeric)

#%%
l_multi = [s for s in l if s.isalpha() and not s.startswith('t')]
print(l_multi)

l_multi_or = [s for s in l if (s.isalpha() and not s.startswith('t')) or ('bbb' in s)]
print(l_multi_or)

#%%
import re
l = ['oneXXXaaa', 'twoXXXbbb', 'three999aaa', '000111222']
l_re_match = [s for s in l if re.match('.*XXX.*', s)]
print(l_re_match)

l_re_sub_all = [re.sub('(.*)XXX(.*)', r'\2---\1', s) for s in l]
print(l_re_sub_all)

l_re_sub = [re.sub('(.*)XXX(.*)', r'\2---\1', s) for s in l if re.match('.*XXX.*', s)]
print(l_re_sub)

