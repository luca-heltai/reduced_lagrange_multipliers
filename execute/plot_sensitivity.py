#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:26:02 2024

@author: camilla
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tikzplotlib

SAVE = True

os.chdir('/home/camilla/Desktop/COUPLING/')

directory = 'output/sensitivity_inclusions_2/'
df = pd.read_csv(directory+'result_execute_sensitivity.csv')
df.head()

fig, axs = plt.subplots(2, 2, layout='constrained')
#hList = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
refList = [5, 10, 20, 50, 100]
hList = [0.5, 0.05, 0.01, 0.005, 0.001]

abscissa = 'ref'
selection = 'h'
print(str(hList[0]))
for c_h in hList:
    x=np.asarray(df[df[selection] == c_h][abscissa])
                            
    axs[0,0].loglog(x, -np.asarray(df[df[selection] == c_h]['p0']), marker = 'o', linestyle='None')
    axs[0,0].set_title('Vessel 0')
    axs[0,0].set_xlabel("transversal refinement")
    
    axs[0,1].loglog(x,-np.asarray(df[df[selection] == c_h]['p1']), marker = 'o', linestyle='None')
    axs[0,1].set_title('Vessel 1')
    axs[0,1].set_xlabel("transversal refinement")
    
    axs[1,0].loglog(x,-np.asarray(df[df[selection] == c_h]['p2']), marker = 'o', linestyle='None', label=c_h)
    axs[1,0].set_title('Vessel 2')
    axs[1,0].set_xlabel("transversal refinement")

 #   axs[0,0].legend(str(c_h), loc='upper right')
  #  print(str(c_h))
fig.legend(loc='lower center')
fig.suptitle('- External Pressure')
    

if SAVE:
    tikzplotlib.save("/home/camilla/Desktop/latex/Latex/ECCOMAS24/pres/tikzsource/transversal.tikz")
else:
    plt.show()
        

fig2, axs2 = plt.subplots(2, 2, layout='constrained')

abscissa = 'h'
selection = 'ref'

for c_h in refList:
    x=np.asarray(df[df[selection] == c_h][abscissa])
                            
    axs2[0,0].loglog(x, -np.asarray(df[df[selection] == c_h]['p0']), marker = 'o')
    axs2[0,0].set_title('Vessel 0')
    axs2[0,0].set_xlabel("longitudinal refinement")
    
    axs2[0,1].loglog(x, -np.asarray(df[df[selection] == c_h]['p1']), marker = 'o')
    axs2[0,1].set_title('Vessel 1')
    axs2[0,1].set_xlabel("longitudinal refinement")
    
    axs2[1,0].loglog(x, -np.asarray(df[df[selection] == c_h]['p2']), marker = 'o', label=c_h)
    axs2[1,0].set_title('Vessel 2')
    axs2[1,0].set_xlabel("longitudinal refinement")
    

    
    fig2.legend(loc='lower center')
    fig2.suptitle('-External Pressure')
    

if SAVE:
    tikzplotlib.save("/home/camilla/Desktop/latex/Latex/ECCOMAS24/pres/tikzsource/longitudinal.tikz")
else:
    plt.show()

# fig2, axs2 = plt.subplots(2, 2, layout='constrained')

# for c_h in refList:
#     x=np.asarray(df[df[selection] == c_h][abscissa])
                            
#     axs2[0,0].loglog(-np.asarray(df[df[selection] == c_h]['p0']), x, marker = 'o')
#     axs2[0,0].set_title('Vessel 0')
#     axs2[0,0].set_ylabel(r"longitudinal refinement")
    
#     axs2[0,1].loglog(-np.asarray(df[df[selection] == c_h]['p1']), x, marker = 'o')
#     axs2[0,1].set_title('Vessel 1')
#     axs2[0,1].set_ylabel(r"longitudinal refinement")
    
#     axs2[1,0].loglog(-np.asarray(df[df[selection] == c_h]['p2']), x, marker = 'o')
#     axs2[1,0].set_title('Vessel 2')
#     axs2[1,0].set_ylabel(r"longitudinal refinement")

    
#     fig2.legend(loc='lower center')
#     fig2.suptitle('-External Pressure')
    

# if SAVE:
#     plt.savefig(directory+"longit_2.pdf")
# else:
#     plt.show()













































































































# vessel_pressure = 'p1'
# #current_ref = [5, 10, 20, 50, 100]

# #for ref in current_ref:
# x=np.asarray(df[df['ref'] == 5]['h'])
# y=np.asarray(df[df['ref'] == 5][vessel_pressure])

# plt.plot(x, y, color = 'g', linestyle = 'dashed')

# x=np.asarray(df[df['ref'] == 10]['h'])
# y=np.asarray(df[df['ref'] == 10][vessel_pressure])

# plt.plot(x, y, color = 'blue', linestyle = 'dashed')

# x=np.asarray(df[df['ref'] == 20]['h'])
# y=np.asarray(df[df['ref'] == 20][vessel_pressure])

# plt.plot(x, y, color = 'black', linestyle = 'dashed')

# x=np.asarray(df[df['ref'] == 50]['h'])
# y=np.asarray(df[df['ref'] == 50][vessel_pressure])

# plt.plot(x, y, color = 'red', linestyle = 'dashed')

# x=np.asarray(df[df['ref'] == 100]['h'])
# y=np.asarray(df[df['ref'] == 100][vessel_pressure])

# plt.plot(x, y, color = 'g', linestyle = 'dashed')

# plt.show() 

# #for ref in current_ref:
# x=np.asarray(df[df['h'] == 0.1]['ref'])
# y=np.asarray(df[df['h'] == 0.1][vessel_pressure])

# plt.plot(x, y, color = 'g', linestyle = 'dashed')

# x=np.asarray(df[df['h'] == 0.01]['ref'])
# y=np.asarray(df[df['h'] == 0.01][vessel_pressure])

# plt.plot(x, y, color = 'blue', linestyle = 'dashed')

# x=np.asarray(df[df['h'] == 0.001]['ref'])
# y=np.asarray(df[df['h'] == 0.001][vessel_pressure])

# plt.plot(x, y, color = 'black', linestyle = 'dashed')

# x=np.asarray(df[df['h'] == 0.5]['ref'])
# y=np.asarray(df[df['h'] == 0.5][vessel_pressure])

# plt.plot(x, y, color = 'red', linestyle = 'dashed')

# x=np.asarray(df[df['h'] == 0.05]['ref'])
# y=np.asarray(df[df['h'] == 0.05][vessel_pressure])

# plt.plot(x, y, color = 'g', linestyle = 'dashed')

# x=np.asarray(df[df['h'] == 0.005]['ref'])
# y=np.asarray(df[df['h'] == 0.005][vessel_pressure])

# plt.plot(x, y, color = 'yellow', linestyle = 'dashed')

# plt.show() 
