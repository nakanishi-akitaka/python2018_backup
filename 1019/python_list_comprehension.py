# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-comprehension/
Created on Fri Oct 19 12:11:24 2018

@author: Akitaka
"""
squares = [i**2 for i in range(5)]
print(squares)

squares = [] 
for i in range(5):
    squares.append(i**2)

print(squares)

#%%
odds = [i for i in range(10) if i % 2 == 1]
print(odds)

odds = []
for i in range(10):
    if i % 2 == 1:
        odds.append(i)

print(odds)

#%%
odd_even = ['odd' if i % 2 == 1 else 'even' for i in range(10)]
print(odd_even)

odd_even = []
for i in range(10):
    if i % 2 == 1:
        odd_even.append('odd')
    else:
        odd_even.append('even')

print(odd_even)

#%%
odd10 = [i * 10 if i% 2 == 1 else i for i in range(10)]
print(odd10)


#%%
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

flat = [x for row in matrix for x in row]
print(flat)

flat = []
for row in matrix:
    for x in row:
        flat.append(x)

print(flat)

#%%
cells = [(row, col) for row in range(3) for col in range(2)]
print(cells)

cells = [(row, col) for row in range(3) for col in range(2) if col == row]
print(cells)

cells = [(row, col) for row in range(3) if row % 2 == 0 for col in range(2) if col % 2 == 0]
print(cells)


#%%
l = [i**2 for i in range(5)]
print(l)
print(type(l))

g = (i**2 for i in range(5))
print(g)
print(type(g))

for i in g:
    print(i)

#%%
g_cells = ((row, col) for row in range(0, 3) for col in range(0, 2) if col == row)
print(type(g_cells))
for i in g_cells:
    print(i)

print(sum([i**2 for i in range(5)]))
print(sum((i**2 for i in range(5))))
print(sum(i**2 for i in range(5)))

