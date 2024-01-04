#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:22:31 2021

@author: caiazzo, belponer, sanvido
"""

import sys
sys.path.append("/opt/miniconda3/lib/python3.9/site-packages")
# import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils_1d_to_3d_data import mesh1d


m = mesh1d(vessels='../bifurcation/vess.dat',
           nodes='../bifurcation/nodes.dat')
#h3DList = [0.1, 0.01, 0.001, 0.5, 0.05, 0.005]
h3DList = [0.0001, 0.0005]
area_test_value = 0.001

for h3D in h3DList:
    m = mesh1d(vessels='../bifurcation/vess.dat',
               nodes='../bifurcation/nodes.dat')
    m.discretize(h3D)
    
    all_points = m.get_points()
    all_dir = m.get_directions()
    all_areas = m.get_areas_reference()
    all_vessID = m.get_vessel_ID()
    
    
    if ((len(all_points) == len(all_dir)) & (len(all_points) == len(all_areas))):
        f = open('../data_sensitivity/inclusions_points_bifurcation_'+str(h3D)+'.txt','w')
        g = open('../data_sensitivity/inclusions_data_bifurcation_'+str(h3D)+'.txt','w')
        for k in range(0,len(all_points)):
            # if ((all_points[k][1] <= 1) or (all_points[k][0] > 1 and all_points[k][0] - np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2 > 1) or (all_points[k][0] < 1 and all_points[k][0] + np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2 < 1)):
            #     plt.scatter(all_points[k][0], all_points[k][1], c='black')
            # # else:
            # #     print(all_points[k][0])
            # #     print(all_areas[k])
            # #     print(np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2)
            # else:
            #     if (all_points[k][0] > 1):
            #         plt.scatter(all_points[k][0]+np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2, all_points[k][1] - np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2, c='gray')
            #         plt.scatter(all_points[k][0]-np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2, all_points[k][1] + np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2, c='gray')
            #         plt.scatter(all_points[k][0], all_points[k][1], c='gray')
            #     if (all_points[k][0] < 1): 
            #         plt.scatter(all_points[k][0]+np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2, all_points[k][1] + np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2, c='blue')
            #         plt.scatter(all_points[k][0]-np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2, all_points[k][1] - np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2, c='blue')
            #         plt.scatter(all_points[k][0], all_points[k][1], c='blue')
            
            #if ((all_points[k][1] <= 1) or (all_points[k][0] > 1 and all_points[k][0] - np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2 > 1) or (all_points[k][0] < 1 and all_points[k][0] + np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2 < 1)):
            if (((all_points[k][1] <= 1) or (all_points[k][0] > 1 and all_points[k][0] - np.sqrt(area_test_value/np.pi)*np.sqrt(2)/2 > 1) 
                or (all_points[k][0] < 1 and all_points[k][0] + np.sqrt(area_test_value/np.pi)*np.sqrt(2)/2 < 1))
                and (all_points[k][1] + np.sqrt(area_test_value/np.pi)*np.sqrt(2)/2 < 2)):
    
                f.write(str(all_points[k][0]) + " " + str(all_points[k][1]) + " " + str(all_points[k][2]) + " " +
                    str(all_dir[k][0]) + " " + str(all_dir[k][1]) + " " + str(all_dir[k][2]) + " " + 
                    #str(all_areas[k]) + " " +
                    "0.001" + " " +
                    str(all_vessID[k]) + "\n"  )
    
                g.write("0.0005 0 0 0 0.0005 0 0 0 0 \n") 
               
        f.close()
        g.close()
    
    # plotx1 = 0
    # plotx2 = 2
    # plt.xlim(plotx1,plotx2)
    # plt.ylim(plotx1,plotx2)
    # plt.show()
