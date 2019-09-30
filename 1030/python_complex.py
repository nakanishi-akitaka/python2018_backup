# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-complex/
Created on Tue Oct 30 12:04:10 2018

@author: Akitaka
"""
c = 3 + 4j

print(c)
print(type(c))
# (3+4j)
# <class 'complex'>

# c = 3 + j
# NameError: name 'j' is not defined

c = 3 + 1j

print(c)
# (3+1j)

c = 3j

print(c)
# 3j

c = 3 + 0j

print(c)
# (3+0j)

c = 1.2e3 + 3j

print(c)
# (1200+3j)

c = complex(3, 4)

print(c)
print(type(c))
# (3+4j)
# <class 'complex'>

#%%
c = 3 + 4j

print(c.real)
print(type(c.real))
# 3.0
# <class 'float'>

print(c.imag)
print(type(c.imag))
# 4.0
# <class 'float'>

# c.real = 5.5
# AttributeError: readonly attribute

#%%
c = 3 + 4j

print(c.conjugate())
# (3-4j)

#%%
c = 3 + 4j

print(abs(c))
# 5.0

c = 1 + 1j

print(abs(c))
# 1.4142135623730951

#%%
import cmath
import math

c = 1 + 1j

print(math.atan2(c.imag, c.real))
# 0.7853981633974483

print(cmath.phase(c))
# 0.7853981633974483

print(cmath.phase(c) == math.atan2(c.imag, c.real))
# True

print(math.degrees(cmath.phase(c)))
# 45.0

#%%
c = 1 + 1j

print(cmath.polar(c))
print(type(cmath.polar(c)))
# (1.4142135623730951, 0.7853981633974483)
# <class 'tuple'>

print(cmath.polar(c)[0] == abs(c))
# True

print(cmath.polar(c)[1] == cmath.phase(c))
# True

print(cmath.rect(1, 1))
# (0.5403023058681398+0.8414709848078965j)

print(cmath.rect(1, 0))
# (1+0j)

print(cmath.rect(cmath.polar(c)[0], cmath.polar(c)[1]))
# (1.0000000000000002+1j)

r = 2
ph = math.pi

print(cmath.rect(r, ph).real == r * math.cos(ph))
# True

print(cmath.rect(r, ph).imag == r * math.sin(ph))
# True

#%%
c1 = 3 + 4j
c2 = 2 - 1j

print(c1 + c2)
# (5+3j)

print(c1 - c2)
# (1+5j)

print(c1 * c2)
# (10+5j)

print(c1 / c2)
# (0.4+2.2j)

print(c1 ** 3)
# (-117+44j)

print((-3 + 4j) ** 0.5)
# (1.0000000000000002+2j)

print((-1) ** 0.5)
# (6.123233995736766e-17+1j)

print(cmath.sqrt(-3 + 4j))
# (1+2j)

print(cmath.sqrt(-1))
# 1j

print(c1 + 3)
# (6+4j)

print(c1 * 0.5)
# (1.5+2j)
