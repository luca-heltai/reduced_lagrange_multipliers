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
#h3DList = [0.0001, 0.0005]
h3DList = [0.05]
area_test_value = 0.0001

x1 = 1.8
sin_alpha = np.abs(x1-1)/np.sqrt((x1-1)**2+1)
cos_alpha = 1/np.sqrt((x1-1)**2+1)

for h3D in h3DList:
    m = mesh1d(vessels='../bifurcation/vess.dat',
               nodes='../bifurcation/nodes.dat')
    m.discretize(h3D)
    
    all_points = m.get_points()
    all_dir = m.get_directions()
    all_areas = m.get_areas_reference()
    all_vessID = m.get_vessel_ID()
    
    
    if ((len(all_points) == len(all_dir)) & (len(all_points) == len(all_areas))):
        f = open('../inclusions_points_bif_2503_'+str(h3D)+'.txt','w')
        g = open('../inclusions_data_bif_2503_'+str(h3D)+'.txt','w')
        for k in range(0,len(all_points)):
            # if ((all_points[k][1] <= 1) or (all_points[k][0] > 1 and all_points[k][0] - np.sqrt(area_test_value/np.pi)*cos_alpha > 1) 
            #     or (all_points[k][0] < 1 and all_points[k][0] + np.sqrt(area_test_value/np.pi)*cos_alpha < 1)):
            plt.scatter(all_points[k][0], all_points[k][1], c='black')
            # # else:
            # #     print(all_points[k][0])
            # #     print(all_areas[k])
            # #     print(np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2)
            # else:
            #     print(all_points[k][1])
            #     print(all_points[k][0], all_points[k][1], all_points[k][0] - np.sqrt(area_test_value/np.pi)*np.sqrt(2)/2) 
            #     print(all_points[k][0], all_points[k][1], all_points[k][0] + np.sqrt(area_test_value/np.pi)*np.sqrt(2)/2)
            #     if (all_points[k][0] > 1):
            #         plt.scatter(all_points[k][0]+np.sqrt(area_test_value/np.pi)*cos_alpha, all_points[k][1] - np.sqrt(area_test_value/np.pi)*sin_alpha, c='gray')
            #         plt.scatter(all_points[k][0]-np.sqrt(area_test_value/np.pi)*cos_alpha, all_points[k][1] + np.sqrt(area_test_value/np.pi)*sin_alpha, c='gray')
            #         plt.scatter(all_points[k][0], all_points[k][1], c='gray')
            #     if (all_points[k][0] < 1): 
            #         plt.scatter(all_points[k][0]+np.sqrt(area_test_value/np.pi)*cos_alpha, all_points[k][1] + np.sqrt(area_test_value/np.pi)*sin_alpha, c='blue')
            #         plt.scatter(all_points[k][0]-np.sqrt(area_test_value/np.pi)*cos_alpha, all_points[k][1] - np.sqrt(area_test_value/np.pi)*sin_alpha, c='blue')
            #         plt.scatter(all_points[k][0], all_points[k][1], c='blue')
            
            # #if ((all_points[k][1] <= 1) or (all_points[k][0] > 1 and all_points[k][0] - np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2 > 1) or (all_points[k][0] < 1 and all_points[k][0] + np.sqrt(all_areas[k]/np.pi)*np.sqrt(2)/2 < 1)):
            # if (((all_points[k][1] <= 1) or (all_points[k][0] > 1 and all_points[k][0] - np.sqrt(area_test_value/np.pi)*cos_alpha > 1) 
            #     or (all_points[k][0] < 1 and all_points[k][0] + np.sqrt(area_test_value/np.pi)*cos_alpha < 1))
            #     and (all_points[k][1] + np.sqrt(area_test_value/np.pi)*np.sqrt(2)/2 < 2)):
    
            f.write(str(all_points[k][0]) + " " + str(all_points[k][1]) + " " + str(all_points[k][2]) + " " +
                    str(all_dir[k][0]) + " " + str(all_dir[k][1]) + " " + str(all_dir[k][2]) + " " + 
                    #str(all_areas[k]) + " " +
                    str(np.sqrt(area_test_value/np.pi)) + " " + 
                    str(all_vessID[k]) + "\n"  )
    
            g.write("0.01 0 0 0 0.01 0 0 0 0 \n") 
               
        f.close()
        g.close()
    
    plotx1 = 0
    plotx2 = 2
    plt.xlim(plotx1,plotx2)
    plt.ylim(plotx1,plotx2)
    plt.show()
