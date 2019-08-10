# -*- coding: utf-8 -*-
"""
Convert Chemical formula -> atomic number & number of atoms 
Created on Fri Jun 22 11:12:16 2018

@author: Akitaka
"""
print(__doc__)

# modules
from pymatgen import Composition
import numpy as np
import pandas as pd
#
# convert Chemical formula -> atomic number & number of atoms 
# read from csv
name = 'tmxo2_gap.csv'
data = np.array(pd.read_csv(name))[:,:]
y=data[:,0]
X=data[:,1:]
test=[]
test_cls=[]
for i in range(len(y)):
    material = Composition(y[i])
    atomicNo = []
    natom = []
    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))
    yx = atomicNo+natom+list(X[i])
    if(X[i,2] > 0):
        X[i,2] = 2
        yx_cls = atomicNo+natom+list(X[i])
    else:
        X[i,2] = 1
        yx_cls = atomicNo+natom+list(X[i])
#   {0[8]}=gap, {0[0]}, {0[1]}, {0[2]} = anomic No, {0[3]}, {0[4]}, {0[5]} = # of atom
#    print('{0[8]}, {0[0]}, {0[1]}, {0[2]}, {0[3]}, {0[4]}, {0[5]}'.format(yx))
#   {0[8]}=gap, {0[0]}, {0[1]} = anomic No of A and B (ABO2)
#    print('{0[8]}, {0[0]}, {0[1]}'.format(yx))
    test.append(yx)
    test_cls.append(yx_cls)
test = pd.DataFrame(test)
test.columns = ['ZA','ZB','ZC','NA','NB','NC','Ehul','Sgrp','Eg','m*']
test.to_csv('test1.csv',index=False)
test_cls = pd.DataFrame(test_cls)
test_cls.columns = ['ZA','ZB','ZC','NA','NB','NC','Ehul','Sgrp','I2M1','m*']
test_cls.to_csv('test1_cls.csv',index=False)

name = 'test1.csv'
data = np.array(pd.read_csv(name))[:,:]
# print(data[0:5])
