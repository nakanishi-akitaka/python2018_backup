# -*- coding: utf-8 -*-
"""
Calculate parameters from physical properties

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

# import math
import numpy as np
from pymatgen import Element
fe=Element("Fe")
for x in "C": # "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr":
    print(Element(x),Element(x).atomic_mass)
    print(Element(x),Element(x).atomic_radius)
    print(Element(x),Element(x).density_of_solid)
    print(Element(x),Element(x).melting_point)
    print(Element(x),Element(x).thermal_conductivity)
    print(Element(x),Element(x).valence)

from pymatgen import Composition
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
    print("{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f},\
 {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
 mu, nu, gm, wg, en, we, ra, wr, sd, ws))
    return # mu, nu, gm, wg, en, we, ra, wr, sd, ws

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
therm = []
amass = []
meltp = []
aradi = []
dense = []
# ex: Re6Zr1
material = Composition("Re6Zr")
for element in material:
    natom.append(material.get_atomic_fraction(element)*material.num_atoms)
    therm.append(float(str(element.thermal_conductivity).split("W")[0]))
    amass.append(float(str(element.atomic_mass).split("a")[0]))
    meltp.append(float(str(element.melting_point).split("K")[0]))
    aradi.append(float(str(element.atomic_radius).split("a")[0]))
    dense.append(float(str(element.density_of_solid).split("k")[0]))

#print(therm)
#print(amass)
#print(meltp)
#print(aradi)
#print(dense)
calc_parameters(*natom, *therm)
calc_parameters(6,1,48,23)
calc_parameters(*natom, *amass)
calc_parameters(6,1,186.207, 91.224)
calc_parameters(*natom, *meltp)
calc_parameters(6,1,3459.0, 2128.0)
calc_parameters(*natom, *aradi)
calc_parameters(6,1,1.35, 1.55)
calc_parameters(*natom, *dense)
calc_parameters(6,1,21020.0, 6511.0)

#%%
# use mendeleev
from mendeleev import element


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
calc_parameters(*natom, *amass)
calc_parameters(*natom, *eion1)
calc_parameters(*natom, *aradi)
calc_parameters(*natom, *dense)
calc_parameters(*natom, *elaff)
calc_parameters(*natom, *meltp)
calc_parameters(*natom, *therm)
calc_parameters(*natom, *nvale)
