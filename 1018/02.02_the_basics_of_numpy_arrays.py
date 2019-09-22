# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html
Created on Thu Oct 18 16:35:13 2018

@author: Akitaka
"""

# NumPy Array Attributes
import numpy as np
np.random.seed(0)
x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3, 4))
x3 = np.random.randint(10, size=(3, 4, 5))
print("x3 ndim:  ", x3.ndim)
print("x3 shape: ", x3.shape)
print("x3 size:  ", x3.size)
print("x3 dtype: ", x3.dtype)
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")

# Array Indexing: Accessing Single Elements

print()
print(x1)
print(x1[0])
print(x1[4])
print(x1[-1])
print(x1[-2])

print()
print(x2)
print(x2[0, 0])
print(x2[2, 0])
print(x2[2, -1])
x2[0, 0] = 12
print(x2)
x1[0] = 3.14159
print(x1)

# Array Slicing: Accessing Subarrays
x = np.arange(10)
print(x)

print(x[:5])
print(x[5:])
print(x[4:7])
print(x[::2])
print(x[1::2])
print(x[::-1])
print(x[5::-2])

## Multi-dimensional subarrays

print(x2)
print(x2[:2, :3])  # two raws, three columns
print(x2[:3, ::2])  # all rows, every other column
print(x2[::-1, ::-1])

### Accessing array rows and columns
print(x2[:, 0])  # first column of x2
print(x2[0, :])  # first row of x2
print(x2[0])  # equivalent to x2[0, :]

## Subarrays as no-copy views
print(x2)
x2_sub = x2[:2, :2]
print(x2_sub)

x2_sub[0, 0] = 99
print(x2_sub)
print(x2)

## Creating copies of arrays
x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)
x2_sub_copy[0, 0] = 42
print(x2_sub_copy)
print(x2)

# Reshaping of Arrays
grid = np.arange(1, 10).reshape((3, 3))
print(grid)

x = np.array([1, 2, 3])

# row vector via reshape
print(x.reshape((1, 3)))

# row vector via newaxis
print(x[np.newaxis, :])

# column vector via reshape
print(x.reshape((3, 1)))

# column vector via newaxis
print(x[:, np.newaxis])


## Concatenation of arrays
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
print(np.concatenate([x, y]))
z = [99, 99, 99]
print(np.concatenate([x, y, z]))

grid = np.array([[1, 2, 3], [4, 5, 6]])
# concatenate along the first axis
print(np.concatenate([grid, grid]))

# concatenate along the second axis (zero-indexed)
print(np.concatenate([grid, grid], axis=1))

x = np.array([1, 2, 3])
grid = np. array([[9, 8, 7], [6, 5, 4]])

# vertically stack the arrays
print(np.vstack([x, grid]))

# horizontally stack the arrays
y = np.array([[99],[99]])
print(np.hstack([grid, y]))


## Splitting of arrays
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

grid = np.arange(16).reshape((4, 4))
print(grid)

upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)

left, right = np.hsplit(grid, [2])
print(left)
print(right)
