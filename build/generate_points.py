#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:54:58 2022

@author: belponer
"""

# seed the pseudorandom number generator
from random import seed
#from random import random
from random import uniform
# seed random number generator
seed(1)
# generate some random numbers

f = open('inclusion_points.txt','w')
k = 0
radius = 0.05
while k < 50:
    x = uniform(-1+radius, 1-radius)
    y = uniform(-1+radius, 1-radius)
    f.write(str(x) + " " + str(y) + " " + str(radius)+" \n"  )
    #print(str(x) + " " + str(y) + ", r = " + str(r) + " \n")
    k=k+1
f.close()