# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/02.03-computation-on-arrays-ufuncs.html
Created on Thu Oct 18 20:43:51 2018

@author: Akitaka
"""
# The Slowness of Loops
import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1, 10, size=5)
print(compute_reciprocals(values))

big_array = np.random.randint(1, 100, size = 100)
compute_reciprocals(big_array)

# Introducing UFuncs
print(compute_reciprocals(values))
print(1.0 / values)
print(np.arange(5) / np. arange(1, 6))
x = np.arange(9).reshape((3, 3))
print(2 ** x)

# Exploring NumPy's UFuncs
## Array arithmetic
x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)

print("-x     = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)

print(-(0.5*x + 1) ** 2)

print(np.add(x, 2))

## Absolute value

x = np.array([-2, -1, 0, 1, 2])
print(abs(x))
print(np.absolute(x))
print(np.abs(x))

x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
print(np.abs(x))

## Trigonometric functions
theta = np.linspace(0, np.pi, 3)
print("thetha      = ", theta)
print("sin(thetha) = ", np.sin(theta))
print("cos(thetha) = ", np.cos(theta))
print("tan(thetha) = ", np.tan(theta))

x = [-1, 0, 1]
print("x         = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))

## Exponents and logarithms
x = [1, 2, 3]
print("x   =", x)
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("3^x =", np.power(3, x))

x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))

x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 = ", np.expm1(x))
print("log(1 + x) = ", np.log1p(x))

## Specialized ufuncs
from scipy import special
# Gamma functions (generalized factrials) and related functions
x = [1, 5, 10]
print("gamma(x)     =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2)   =", special.beta(x, 2))

# Error function (integral of Gaussian)
# its compement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x)    =", special.erf(x))
print("erfc(x)   =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))

#%%
# Advanced Ufunc Features
## Specifying output

x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)

y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)

## Aggregates
x = np.arange(1, 6)
print(np.add.reduce(x))
print(np.multiply.reduce(x))
print(np.add.accumulate(x))
print(np.multiply.accumulate(x))

## Outer products
x = np.arange(1, 6)
print(np.multiply.outer(x, x))

