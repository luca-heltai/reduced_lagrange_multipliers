# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:00:47 2023

@author: sarah, camillabelponer
"""
import numpy  as np
import matplotlib.pyplot as plt
import os
#import matplotlib2tikz
import tikzplotlib

os.chdir('/home/camilla/Desktop/COUPLING/')
directory=os.getcwd()+"/output/bifurcation_200521/"


# #testcase="control_elastic"
# #testcase="uncoupled_elastic"
# testcase="brain_elastic"
# #testcase="liver_elastic"
# dataDirectory=directory+testcase+"/1D/"

# mmHg=1333.22
# # fixed parameters
# dt = "0.001"
# kmref = "1"
# fig, axs = plt.subplots(2, 2, layout='constrained')

# numVess=["1", "2", "3"]

# for j in range(len(numVess)):

#     PExt_file=np.loadtxt(dataDirectory+"PresEXTVess3D_"+numVess[j]+"Cell0_k_0dt"+dt+"_omega_1DXmax1kmRef"+kmref+".txt", skiprows=8000)
#     Pres_file=np.loadtxt(dataDirectory+"PresVess3D_"+numVess[j]+"Cell0_k_0dt"+dt+"_omega_1DXmax1kmRef"+kmref+".txt", skiprows=8000)
#     Area_file=np.loadtxt(dataDirectory+"areaVess3D_"+numVess[j]+"Cell0_k_0dt"+dt+"_omega_1DXmax1kmRef"+kmref+".txt", skiprows=8000)
#     Flux_file=np.loadtxt(dataDirectory+"qVess3D_"+numVess[j]+"Cell0_k_0dt"+dt+"_omega_1DXmax1kmRef"+kmref+".txt", skiprows=8000)

                            
#     axs[0,0].plot(PExt_file[:,0], PExt_file[:,1]/mmHg)
#     axs[0,0].set_title('External Pressure pe')
#     axs[0,0].set_xlabel("time [s]")
#     axs[0,0].set_ylabel(r"External Pressure [mmHg]")      

#     axs[0, 1].plot(Pres_file[:,0], Pres_file[:,1]/mmHg)
#     axs[0, 1].set_title('Pressure p')
#     axs[0,1].set_xlabel("time [s]")
#     axs[0,1].set_ylabel(r"Pressure [mmHg]")     
    
#     axs[1, 0].plot(Area_file[:,0], Area_file[:,1])
#     #axs[1, 0].vlines(np.linspace(0.1,1,9), np.min(Area_file[:,1]), np.max(Area_file[:,1]), colors = 'black', linewidth = 0.5)
#     axs[1, 0].set_title(r'Area a')
#     axs[1,0].set_xlabel("time [s]")
#     axs[1,0].set_ylabel("Area [cm^2]")
# #    axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# #axs[1,0].legend()

#     axs[1, 1].plot(Flux_file[:,0], Flux_file[:,1], label="numVess"+numVess[j])
#     #axs[1, 1].vlines(np.arange(0.1, 1, 0.1), np.min(Flux_file[:,1]), np.max(Flux_file[:,1]), colors = 'black', linewidth = 0.5)
#     axs[1, 1].set_title('Flow')
#     axs[1,1].set_xlabel("time [s]")
#     axs[1,1].set_ylabel("flow [ml/s]")
    
    
#     fig.legend(loc='lower center')
#     fig.suptitle(testcase)

# #plt.savefig(directory+testcase+"graphs.pdf")

# plt.show()



###############################################################################
os.chdir('/home/camilla/Desktop/COUPLING/')
directory=os.getcwd()+"/output/bifurcation_b/"


# fixed parameters
mmHg=1333.22
fig, axs = plt.subplots(2, 2, layout='constrained')

numVess=["1", "2", "3"]
#cases=["control_elastic", "uncoupled_elastic", "brain_elastic"]
cases=["uncoupled_elastic", 
       #"liver_elastic", 
       "brain_elastic"
       ]
j = 2
cellnr="0"

for testcase in cases:
    
    dataDirectory=directory+testcase+"_mode0/1D/"

    PExt_file=np.loadtxt(dataDirectory+"PresEXTVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
    Pres_file=np.loadtxt(dataDirectory+"PresVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
    Area_file=np.loadtxt(dataDirectory+"areaVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
    Flux_file=np.loadtxt(dataDirectory+"qVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)

                            
    axs[0,0].plot(PExt_file[:,0], PExt_file[:,1]/mmHg)
    axs[0,0].set_title('External Pressure pe')
    axs[0,0].set_xlabel("time [s]")
    axs[0,0].set_ylabel(r"External Pressure [mmHg]")      

    axs[0, 1].plot(Pres_file[:,0], Pres_file[:,1]/mmHg)
    axs[0, 1].set_title('Pressure p')
    axs[0,1].set_xlabel("time [s]")
    axs[0,1].set_ylabel(r"Pressure [mmHg]")     
    
    axs[1, 0].plot(Area_file[:,0], Area_file[:,1])
    #axs[1, 0].vlines(np.linspace(0.1,1,9), np.min(Area_file[:,1]), np.max(Area_file[:,1]), colors = 'black', linewidth = 0.5)
    axs[1, 0].set_title(r'Area a')
    axs[1,0].set_xlabel("time [s]")
    axs[1,0].set_ylabel("Area [cm^2]")
#    axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#axs[1,0].legend()

    axs[1, 1].plot(Flux_file[:,0], Flux_file[:,1], label=testcase)
    #axs[1, 1].vlines(np.arange(0.1, 1, 0.1), np.min(Flux_file[:,1]), np.max(Flux_file[:,1]), colors = 'black', linewidth = 0.5)
    axs[1, 1].set_title('Flow')
    axs[1,1].set_xlabel("time [s]")
    axs[1,1].set_ylabel("flow [ml/s]")
    
    
    fig.legend(loc='lower center')
    fig.suptitle(testcase)

plt.savefig(directory+testcase+"graphs.pdf")

#plt.show()




###############################################################################
os.chdir('/home/camilla/Desktop/COUPLING/')
directory=os.getcwd()+"/output/bifurcation_b/"


# fixed parameters
mmHg=1333.22
fig, axs = plt.subplots(1, 1, layout='constrained')

numVess=["1", "2", "3"]
#cases=["control_elastic", "uncoupled_elastic", "brain_elastic"]
cases=["uncoupled_elastic", 
       #"liver_elastic", 
       "brain_elastic"
       ]
j = 2
cellnr="0"

for testcase in cases:
    
    dataDirectory=directory+testcase+"_mode0/1D/"

    PExt_file=np.loadtxt(dataDirectory+"PresEXTVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
    Pres_file=np.loadtxt(dataDirectory+"PresVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
    Area_file=np.loadtxt(dataDirectory+"areaVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
    Flux_file=np.loadtxt(dataDirectory+"qVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)

                            
    axs.plot(PExt_file[:,0], PExt_file[:,1]/mmHg)
    axs.set_title('External Pressure pe')
    axs.set_xlabel("time [s]")
    axs.set_ylabel(r"External Pressure [mmHg]")      

    
    fig.legend(loc='lower center')


#plt.show()
tikzplotlib.save("/home/camilla/Desktop/latex/Latex/ECCOMAS24/pres/brainPe.tex")


