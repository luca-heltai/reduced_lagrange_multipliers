#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jan 22 11:22:31 2021

@author: caiazzo, belponer, sanvido
"""

import sys
sys.path.append("/opt/miniconda3/lib/python3.9/site-packages")
from utils_1d_to_3d_data import mesh1d
#import matplotlib.pyplot as plt
#import numpy as np

m = mesh1d(vessels='../T_junction/vess.dat',
           nodes='../T_junction/nodes.dat')
h3D = 0.01
m.discretize(h3D)

all_points = m.get_points()
all_dir = m.get_directions()
all_areas = m.get_areas_reference()
all_vessID = m.get_vessel_ID()

print(len(all_points))
print(len(all_dir))
print(len(all_areas))
print(len(all_vessID))

if ((len(all_points) == len(all_dir)) & (len(all_points) == len(all_areas))):
    f = open('../inclusions_points_T.txt','w')
    g = open('../inclusions_data_T.txt','w')
    for k in range(0,len(all_points)):
        f.write(str(all_points[k][0]) + " " + str(all_points[k][1]) + " " + str(all_points[k][2]) + " " +
                str(all_dir[k][0]) + " " + str(all_dir[k][1]) + " " + str(all_dir[k][2]) + " " + 
                #str(all_areas[k]) + " "
                "0.05 " 
                + str(all_vessID[k]) + "\n"  )
        g.write("0.02 0 0 0 0.02 0 0 0 0 \n") 

           
    f.close()
    g.close()
