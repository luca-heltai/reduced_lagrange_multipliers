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

#os.chdir('/home/camilla/Desktop/COUPLING/')
#directory=os.getcwd()+"/output/bifurcation_270524/step3/"


###############################################################################
# plot all vessels together
###############################################################################

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
# plot all quantities together
###############################################################################

save=True

os.chdir('/home/camilla/Desktop/COUPLING/')
directory=os.getcwd()+"/output/bifurcation_280524/"

# fixed parameters
mmHg=1333.22
fig, axs = plt.subplots(2, 2, layout='constrained')

numVess=["1", "2", "3"]
# cases=[#"uncoupled_0_elastic_NCELL1", 
#        #"uncoupled_elastic_NCELL1_k9",
#        #"uncoupled_elastic_NCELL1_k8",
#        #"uncoupled_elastic_NCELL1_k75", 
#        #"brain_elastic_mode0_NCELL1",
#        #"liver_elastic_mode0_NCELL1",
#        #"uncoupled_0_elastic_NCELL2", 
#        #"uncoupled_elastic_NCELL2_k8", 
#        #"brain_elastic_mode0_NCELL2",
#        #"liver_elastic_mode0_NCELL2", # same as with 1 cell
#        #"brain_elastic_mode1_NCELL3",
#        #"liver_elastic_mode1_NCELL3"
#        "uncoupled_0_elastic",
#        "uncoupled_elastic",
#        "liver_elastic"
#        ]
# j = 0
# cellnr="0"

# for testcase in cases:
    
#     dataDirectory=directory+testcase+"/1D/"

#     PExt_file=np.loadtxt(dataDirectory+"PresEXTVess3D_"+numVess[j]+"Cell"+cellnr+".txt" ,max_rows=3000)
#     Pres_file=np.loadtxt(dataDirectory+"PresVess3D_"+numVess[j]+"Cell"+cellnr+".txt", max_rows=3000)
#     Area_file=np.loadtxt(dataDirectory+"areaVess3D_"+numVess[j]+"Cell"+cellnr+".txt", max_rows=3000)
#     Flux_file=np.loadtxt(dataDirectory+"qVess3D_"+numVess[j]+"Cell"+cellnr+".txt", max_rows=3000)
#     if (testcase == "uncoupled_0_elastic"):
#         PExt_file=np.loadtxt(dataDirectory+"PresEXTVess3D_"+numVess[j]+"Cell"+cellnr+".txt" ,skiprows=8000)
#         Pres_file=np.loadtxt(dataDirectory+"PresVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
#         Area_file=np.loadtxt(dataDirectory+"areaVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
#         Flux_file=np.loadtxt(dataDirectory+"qVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
        

                            
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

#     axs[1, 1].plot(Flux_file[:,0], Flux_file[:,1], label=testcase)
#     #axs[1, 1].vlines(np.arange(0.1, 1, 0.1), np.min(Flux_file[:,1]), np.max(Flux_file[:,1]), colors = 'black', linewidth = 0.5)
#     axs[1, 1].set_title('Flow')
#     axs[1,1].set_xlabel("time [s]")
#     axs[1,1].set_ylabel("flow [ml/s]")
    
    
#     fig.legend(loc='lower center')
#     fig.suptitle(testcase)

# if(save):
#     plt.savefig(directory+"liver_vess"+numVess[j]+".pdf")

# plt.show()




###############################################################################
# plot all quantities divided and save .tex
###############################################################################

show = False

cases=["uncoupled_elastic",
        "liver_elastic"
        ]

organ="liver"
variable=["Pe", "P", "area", "flow"]
cellnr="0"
j=0

for var in variable:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for testcase in cases:
        
        dataDirectory=directory+testcase+"/1D/"
        
        if (var == "Pe"):
            data=np.loadtxt(dataDirectory+"PresEXTVess3D_"+numVess[j]+"Cell"+cellnr+".txt")
            plt.plot(data[:,0], data[:,1]/mmHg)
        if (var == "P"):
            data=np.loadtxt(dataDirectory+"PresVess3D_"+numVess[j]+"Cell"+cellnr+".txt")
            plt.plot(data[:,0], data[:,1]/mmHg)
        if (var == "area"):
            data=np.loadtxt(dataDirectory+"areaVess3D_"+numVess[j]+"Cell"+cellnr+".txt")
            plt.plot(data[:,0], data[:,1])
        if (var == "flow"):
            data=np.loadtxt(dataDirectory+"qVess3D_"+numVess[j]+"Cell"+cellnr+".txt")
            plt.plot(data[:,0], data[:,1])
    
    
    
    if (var == "Pe"):
        ax.set_title('External Pressure [mmHg]')
        ax.set_xlabel("time [s]")
    
    if (var == "P"):
        data=np.loadtxt(directory+"uncoupled_0_elastic/1D/PresVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
        plt.plot(data[:,0], data[:,1]/mmHg)
        ax.set_title('Pressure p')
        ax.set_xlabel("time [s]")  
    
    if (var == "area"):
        data=np.loadtxt(directory+"uncoupled_0_elastic/1D/areaVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
        plt.plot(data[:,0], data[:,1])
        ax.set_title(r'Area a')
        ax.set_xlabel("time [s]")
    
    if (var == "flow"):
        data=np.loadtxt(directory+"uncoupled_0_elastic/1D/qVess3D_"+numVess[j]+"Cell"+cellnr+".txt", skiprows=8000)
        plt.plot(data[:,0], data[:,1])
        ax.set_title('Flow')
        ax.set_xlabel("time [s]")
        
        
    ax.legend(['uncoupled', 'coupling', 'pseudosolver'], loc = 'upper left')
    
    if (show):
        plt.show()
    else:
        tikzplotlib.save("/home/camilla/Desktop/latex/Latex/ECCOMAS24/pres/tikzsource/"+organ+var+numVess[j]+".tikz")




