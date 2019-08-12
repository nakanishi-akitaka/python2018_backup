# -*- coding: utf-8 -*-
"""
Check physical properties of materials

Created on Thu Jul  5 14:04:59 2018

@author: Akitaka

"""

from mendeleev import element
for i in range(1,100):
    x=element(i)
#    print(x.symbol)
#    print(x,x.electron_affinity,"electron affinity")
    if(x.electron_affinity==None):
        print(x,x.electron_affinity,"electron affinity")
    elif(x.electron_affinity<0.0):
        print(x,x.electron_affinity,"electron affinity")
#    if(x.thermal_conductivity==None):
#        print(x,x.thermal_conductivity,"thermal conductivity")

