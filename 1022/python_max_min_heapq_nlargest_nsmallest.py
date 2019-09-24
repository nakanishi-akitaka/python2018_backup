# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-max-min-heapq-nlargest-nsmallest/
Created on Mon Oct 22 16:19:51 2018

@author: Akitaka
"""
l = [3, 6, 7, -1, 23, -10, 18]
print(max(l))
print(min(l))


ld = sorted(l, reverse=True)
print(ld)
print(ld[:3])

la = sorted(l)
print(la)
print(la[:3])

print(sorted(l, reverse=True)[:3])
print(sorted(l)[:3])

print(l)
l.sort(reverse=True)
print(l[:3])
print(l)

l.sort()
print(l[:3])
print(l)


#%%
import heapq
l = [3, 6, 7, -1, 23, -10, 18]
print(heapq.nlargest(3, l))
print(heapq.nsmallest(3, l))
print(l)

