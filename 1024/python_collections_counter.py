# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-collections-counter/
Created on Wed Oct 24 10:41:26 2018

@author: Akitaka
"""
l = ['a', 'a', 'a', 'a', 'b', 'c', 'c']

print(len(l))

l = ['a', 'a', 'a', 'a', 'b', 'c', 'c']

print(l.count('a'))
print(l.count('b'))
print(l.count('c'))
print(l.count('d'))

import collections

c = collections.Counter(l)
print(c)
print(issubclass(type(c),dict))

print(c['a'])
print(c['b'])
print(c['c'])
print(c['d'])

print(c.keys())
print(c.values())
print(c.items())

print(c.most_common())
print(c.most_common()[0])
print(c.most_common()[-1])
print(c.most_common()[0][0])
print(c.most_common()[0][1])
print(c.most_common()[::-1])
print(c.most_common(2))

values, counts = zip(*c.most_common())
print(values)
print(counts)

#%%
print(len(c))
print(set(l))
print(len(set(l)))

#%%
l = list(range(-5, 6))
print(l)
print([i for i in l if i < 0])
print(len([i for i in l if i < 0]))
print([i for i in l if i % 2 == 1])
print(len([i for i in l if i % 2 == 1]))

l = ['apple', 'orange', 'banana']
print([s for s in l if s.endswith('e')])
print(len([s for s in l if s.endswith('e')]))

l = ['a', 'a', 'a', 'a', 'b', 'c', 'c']
print([i for i in l if l.count(i) >= 2])
print(len([i for i in l if l.count(i) >= 2]))

c = collections.Counter(l)
print([i[0] for i in c.items() if i[1] >= 2])
print(len([i[0] for i in c.items() if i[1] >= 2]))

#%%
s = 'government of the people, by the people, for the people.'

s_remove = s.replace(',', '').replace('.', '')
print(s_remove)

word_list = s_remove.split()
print(word_list)

print(word_list.count('people'))
print(len(set(word_list)))
c = collections.Counter(word_list)
print(c)
print(c.most_common()[0][0])


#%%
s = 'supercalifragilisticexpialidocious'
print(s.count('p'))
c = collections.Counter(s)
print(c)
print(c.most_common(5))
values, counts = zip(*c.most_common(5))
print(values)
print(counts)

