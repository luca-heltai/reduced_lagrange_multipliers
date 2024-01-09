#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:53:06 2023

@author: camilla
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax = fig.gca(projection='3d')

data = np.genfromtxt('../old/inclusions_points_bifurcation.txt')
#zero coupon maturity dates
y = data[:,0]
#tenor
x = data[:,1]
#rates
z = data[:,2]

for i in range(0,len(x)-1):
    ax.scatter(float(x[i]), float(y[i]), float(z[i]))
    
plotmin=0
plotmax=2
    
ax.axes.set_xlim(plotmin,plotmax)
ax.axes.set_ylim(plotmin,plotmax)
#ax.axes.set_xlim(0,2)
#ax.axes.set_ylim(0,2)
ax.axes.set_zlim(plotmin,plotmax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_proj_type('ortho')
ax.view_init(elev=0, azim=0) # yz
#ax.view_init(elev=0, azim=90) # xz
#ax.view_init(elev=90, azim=90) # xy
#ax.view_init(elev=0, azim=180) # zy
ax.set_box_aspect(None, zoom=1.25)

plt.show()
