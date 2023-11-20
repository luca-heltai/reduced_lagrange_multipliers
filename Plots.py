# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:00:47 2023

@author: sarah, camillabelponer
"""
import numpy  as np
import matplotlib.pyplot as plt
import os


mmHg=1333.22
directory=os.getcwd()+"/output/mpi_prova/"
dataDirectory=directory+"1D/"

# fixed parameters
dt = "0.01"
fig, axs = plt.subplots(2, 2, layout='constrained')

numVess=["1", "2", "3"]

for j in range(len(numVess)):

    PExt_file=np.loadtxt(dataDirectory+"PresEXTVess3D_"+numVess[j]+"Cell0_k_0dt"+dt+"_omega_1DXmax1kmRef1.txt")
    Pres_file=np.loadtxt(dataDirectory+"PresVess3D_"+numVess[j]+"Cell0_k_0dt"+dt+"_omega_1DXmax1kmRef1.txt")
    Area_file=np.loadtxt(dataDirectory+"areaVess3D_"+numVess[j]+"Cell0_k_0dt"+dt+"_omega_1DXmax1kmRef1.txt")
    Flux_file=np.loadtxt(dataDirectory+"qVess3D_"+numVess[j]+"Cell0_k_0dt"+dt+"_omega_1DXmax1kmRef1.txt")

                            
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

    axs[1, 1].plot(Flux_file[:,0], Flux_file[:,1], label="numVess"+numVess[j])
    #axs[1, 1].vlines(np.arange(0.1, 1, 0.1), np.min(Flux_file[:,1]), np.max(Flux_file[:,1]), colors = 'black', linewidth = 0.5)
    axs[1, 1].set_title('Flow')
    axs[1,1].set_xlabel("time [s]")
    axs[1,1].set_ylabel("flow [ml/s]")
    
    #axs[0,0].plot.xlim(9,10)
    
    fig.legend(loc='lower center')
    



    # axs.plot(x,y, label="vessel"+numVess[j])
    # axs.set_title('External Pressure pe')
    # axs.set_xlabel("time [s]")
    # axs.set_ylabel(r"External Pressure [mmHg]")
    # fig.legend()
    # plt.show()
    # print(j)

plt.savefig(directory+"graphs_b.pdf")

