# -*- coding: utf-8 -*-
"""
Calculate parameters from physical properties
Created on Wed Jul  4 15:39:11 2018

@author: Akitaka

Ref:
"A Data-Driven Statistical Model for Predicting the Critical Temperature
 of a Superconductor"
https://arxiv.org/abs/1803.10260

ref:
20180315test0.py
20180521test2.py
20180426test2.py

1.Atomic Mass atomic, mass units (AMU)
2.First Ionization Energy, kilo-Joules per mole (kJ/mol)
3.Atomic Radius, picometer (pm)
4.Density, kilograms per meters cubed (kg/m3)
5.Electron Affinity, kilo-Joules per mole (kJ/mol)
6.Fusion Heat, kilo-Joules per mole (kJ/mol)
7.Thermal Conductivity, watts per meter-Kelvin (W/(m × K))
8.Valence no units typical number of chemical bonds formed by the element
Table 1: This table shows the properties of an element which are used for creating features to
predict Tc.

Mean = µ = (t1 + t2)/2 
Weighted mean = ν = (p1t1) + (p2t2) 
Geometric mean = (t1t2)**1/2
Weighted geometric mean = (t1)**p1*(t2)**p2
Entropy = −w1 ln(w1) − w2 ln(w2)
Weighted entropy = −A ln(A) − B ln(B) 
Range = t1 − t2 (t1 > t2)
Weighted range = p1 t1 − p2 t2
Standard deviation = [(1/2)((t1 − µ)**2 + (t2 − µ)**2)]**1/2
Weighted standard deviation = [p1(t1 − ν)**2 + p2(t2 − ν)**2)]**1/2
Table 2:

Created on Tue Jul  3 15:37:39 2018

@author: Akitaka
"""

import numpy as np
from pymatgen import Composition
from mendeleev import element

def calc_parameters(n1,n2,t1,t2):
    p1 = n1/(n1+n2)
    p2 = n2/(n1+n2)
    w1 = t1/(t1+t2)
    w2 = t2/(t1+t2)
    A  = (p1*w1)/(p1*w1+p2*w2)
    B  = (p2*w2)/(p1*w1+p2*w2)
    mu = (t1 + t2)/2
    nu = p1 * t1 + p2 * t2
    gm = np.sqrt(t1*t2)
    wg = t1**p1*t2**p2 
    en = -w1*np.log(w1) - w2*np.log(w2)
    we = - A*np.log(A)  -  B*np.log(B)
    ra = np.abs(t1-t2)
    wr = np.abs(p1*t1-p2*t2)
    sd = np.sqrt(((t1-mu)**2+(t2-mu)**2)/2)
    ws = np.sqrt(p1*(t1-nu)**2+p2*(t2-nu)**2)
#    print(p1, p2, w1, w2, A, B)
#    print("{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f},\
# {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
# mu, nu, gm, wg, en, we, ra, wr, sd, ws))
    return mu, nu, gm, wg, en, we, ra, wr, sd, ws
#    Mean = µ = (t1 + t2)/2 
#    Weighted mean = ν = (p1t1) + (p2t2) 
#    Geometric mean = (t1t2)**1/2
#    Weighted geometric mean = (t1)**p1*(t2)**p2
#    Entropy = −w1 ln(w1) − w2 ln(w2)
#    Weighted entropy = −A ln(A) − B ln(B) 
#    Range = t1 − t2 (t1 > t2)
#    Weighted range = p1 t1 − p2 t2
#    Standard deviation = [(1/2)((t1 − µ)**2 + (t2 − µ)**2)]**1/2
#    Weighted standard deviation = [p1(t1 − ν)**2 + p2(t2 − ν)**2)]**1/2
#    
#    print(dir(Element(x)))


natom = []
amass = []
eion1 = []
aradi = []
dense = []
elaff = []
meltp = []
therm = []
nvale = []
# ex: Re6Zr1
material = Composition("Re6Zr")
for x in material:
    natom.append(material.get_atomic_fraction(x)*material.num_atoms)
    x = element(str(x))
    amass.append(x.mass) # AMU
    eion1.append(x.ionenergies[1]) # a dictionary with ionization energies in eV
    aradi.append(x.atomic_radius) # crystal radius in pm
    dense.append(x.density) # g/cm3
    elaff.append(x.electron_affinity) # eV
    meltp.append(x.melting_point) # Kelvin
    therm.append(x.thermal_conductivity) # W /(m K) 
    nvale.append(x.nvalence()) # no unit
#    ref of unit
#    https://mendeleev.readthedocs.io/en/stable/data.html
#    https://mendeleev.readthedocs.io/en/stable/_modules/mendeleev/tables.html
#    https://mendeleev.readthedocs.io/en/stable/code.html
#    1.Atomic Mass atomic, mass units (AMU)
#    2.First Ionization Energy, kilo-Joules per mole (kJ/mol)
#    3.Atomic Radius, picometer (pm)
#    4.Density, kilograms per meters cubed (kg/m3)
#    5.Electron Affinity, kilo-Joules per mole (kJ/mol)
#    6.Fusion Heat, kilo-Joules per mole (kJ/mol)
#    7.Thermal Conductivity, watts per meter-Kelvin (W/(m × K))
#    8.Valence no units typical number of chemical bonds formed by the element

allprm = []
# physical properties
pplist = [amass, eion1, aradi, dense, elaff, meltp, therm, nvale]
for pp in pplist:
    params = calc_parameters(*natom, *pp)
    allprm += list(params)
#    print("{:8.2f}, {:8.2f}, {:8.2f}, {:8.2f}, {:8.2f},\
# {:8.2f}, {:8.2f}, {:8.2f}, {:8.2f}, {:8.2f}".format(*params))

print(allprm)


# read datasets from csv file
# ref: 0627test10.py

import pandas as pd
file = "../python_work_fs01/2018/0330/bandgapDFT.csv"
data = np.array(pd.read_csv(file,header=None))[:,:]
# print(data)
y=data[:,1]
X=data[:,0]
print(X[:5])
print(y[:5])

from time import time
start = time()
Xconv = []
for i in range(5): # range(len(X)):
    material = Composition(X[i])
    natom = []
    amass = []
    eion1 = []
    aradi = []
    dense = []
    elaff = []
    meltp = []
    therm = []
    nvale = []
    for x in material:
        natom.append(material.get_atomic_fraction(x)*material.num_atoms)
        x = element(str(x))
        amass.append(x.mass) # AMU
        eion1.append(x.ionenergies[1]) # a dictionary with ionization energies in eV
        aradi.append(x.atomic_radius_rahm) # Atomic radius by Rahm et al.	pm
        dense.append(x.density) # g/cm3
        elaff.append(x.electron_affinity) # eV
        meltp.append(x.melting_point) # Kelvin
        therm.append(x.thermal_conductivity) # W /(m K) 
        nvale.append(x.nvalence()) # no unit
    allprm = []
    # physical properties
#    pplist = [amass, eion1, aradi, dense, elaff, meltp, therm, nvale]
    pplist = [amass, eion1, aradi, meltp, nvale]
#    print(material,pplist)
    for pp in pplist:
        params = calc_parameters(*natom, *pp)
#        allprm += list(params)    
#    Xconv.append(allprm)
print('{:.2f} seconds '.format(time() - start))

#%%
x=element('Ne')
print(x.electron_affinity)
x.electron_affinity=10.0
print(x.electron_affinity)

#%%
#from mendeleev import element
#for i in range(1,87):
#    x=element(i)
#    print(x.name)
#    print("mas",x.mass) # AMU
#    print("ion",x.ionenergies[1]) # a dictionary with ionization energies in eV
#    print("rad",x.atomic_radius_rahm) # Atomic radius in pm
#    print("den",x.density) # g/cm3
#    print("aff",x.electron_affinity) # eV
#    print("mel",x.melting_point) # Kelvin
#    print("con",x.thermal_conductivity) # W /(m K) 
#    print("val",x.nvalence()) # no unit
