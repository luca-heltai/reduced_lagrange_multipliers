# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:00:47 2023

@author: sarah
"""
import numpy  as np
import matplotlib.pyplot as plt


mmHg=1333.22

#fare plot con 4 grafici, area, flow, pressione, pressione esterna
plot4Media=0 #plot con la pressione media costante
plot4=1#pot usando solve3dk
plotPeConst4= 0  #plot fatti con pressione costante
plotAvsP=0 #plot Area vs Pressione
#fare plot con diverse pressioni esterne/pressioni/area/flow date da k, omega e numVess
plotPek=0
plotpk=0
plotak=0
plotqk=0

parTitle=0 #se mettere titolo o meno (conviene metterlo per plot4Media)





if plotPek:
    #k=["3e+08","4e+08","5e+08", "6e+08"]
    k=["3e+08"]
   # Omega=["1","0.05","0.005","0.0005"]
    Omega=["1"]
    #numVess=["1", "2", "3"]
    numVess=["1"]
    dt=["0.001"]
    cell=["0"]
    
    DXmax=["1"]
    
    network="Bifurcation" # Tree, Bifurcation
    simulation=["elastic"] # pBase, visco, dx, viscodx, elastic
    folder=["Solve3dk"] #or "MeanPressurek", Solve3dk
    
    pBase=["13332.2"]
    
    removeTime=5
    
    nameFig= "PlotPe "+"TimeFrom"+str(removeTime)+network
    for i in range (len(simulation)):
        nameFig=nameFig+"_"+simulation[i]
        if simulation[i]=="pBase":
            for p in range(len(pBase)):
                nameFig=nameFig+"_"+pBase[p]
        if simulation[i]=="dx":
            for dx in range(len(DXmax)):
                nameFig=nameFig+"_"+DXmax[dx]
        
    
    nameFig=nameFig+"k_"
    for i in range (len(k)):
        nameFig=nameFig+"_"+k[i]
        
    nameFig=nameFig+"_Omega"    
    for w in range(len(Omega)):
        nameFig=nameFig+"_"+Omega[w]    
    
    nameFig=nameFig+"_nVess"
    for j in range(len(numVess)):
        nameFig=nameFig+"_"+numVess[j]
        
    nameFig=nameFig+"_dt"
    for j in range(len(dt)):
        nameFig=nameFig+"_"+dt[j]
        
    nameFig=nameFig+"_cell"
    for j in range(len(cell)):
        nameFig=nameFig+"_"+cell[j]
        
    nameFig=nameFig+"_DXmax"
    for j in range(len(DXmax)):
        nameFig=nameFig+"_"+DXmax[j]
        
    
    nameFig=nameFig+".png"
    
    fig, axs = plt.subplots(1, 1, layout='constrained')
    
    for s in range(len(simulation)):
        
    
        for f in range(len(folder)):
        
            for c in range(len(cell)):
            
                for t in range(len(dt)):
                    
                    for i in range(len(k)):
                
                        for w in range(len(Omega)):
                            
                            for j in range(len(numVess)):
                                
                                if removeTime==0:
                                   a=0
                                    
                                else:
                                    if dt[t]=="0.001":
                                        a=int(removeTime*1e3)
                                        
                                    elif dt[t]=="0.0001":
                                        a=int(removeTime*1e4)
                                        
                                    elif dt[t]=="1e-05":
                                        a=int(removeTime*1e5)
                                
                                if (simulation[s]=="elastic"):
                                    
                                    for dx in range(len(DXmax)):
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        
                                        dat = np.loadtxt(solve3DPe)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]/mmHg
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                        
                                    
                                elif (simulation[s]=="pBase"):
                                    for p in range(len(pBase)):
                                    
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"
                                
                                        dat = np.loadtxt(solve3DPe)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]/mmHg
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                
                                elif (simulation[s]=="visco"):
                                    
                                    for dx in range(len(DXmax)):
                                    
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                
                                        dat = np.loadtxt(solve3DPe)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]/mmHg
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                    
                                elif (simulation[s]=="dx"):
                                    for dx in range(len(DXmax)):
                                    
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                
                                        dat = np.loadtxt(solve3DPe)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]/mmHg
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                        
                                elif (simulation[s]=="viscodx"):
                                    for dx in range(len(DXmax)):
                                    
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                
                                        dat = np.loadtxt(solve3DPe)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]/mmHg
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                    
                                    
                
    plt.savefig(nameFig)
    
 
if plotpk:
    #k=["1e+08","2e+08","3e+08"]
    k=["4e+08"]
    #Omega=["1","0.05","0.005","0.0005"]
    Omega=["1"]
    numVess=["1", "2", "3"]
    
    dt=["0.001"]
    cell=["0"]
    
    DXmax=["1"]
    
    network="Bifurcation" # Tree, Bifurcation
    simulation=["elastic"] # pBase, visco, dx, viscodx, elastic
    folder=["Solve3dk"] #or "MeanPressurek", Solve3dk
    
    pBase=[]
    
    removeTime=0
    
    nameFig= "PlotP "+"TimeFrom"+str(removeTime)+network
    
    for i in range (len(simulation)):
        nameFig=nameFig+"_"+simulation[i]
        if simulation[i]=="pBase":
            for p in range(len(pBase)):
                nameFig=nameFig+"_"+pBase[p]
        if simulation[i]=="dx":
            for dx in range(len(DXmax)):
                nameFig=nameFig+"_"+DXmax[dx]
    
    nameFig=nameFig+"k_"
    for i in range (len(k)):
        nameFig=nameFig+"_"+k[i]
        
    nameFig=nameFig+"_Omega"    
    for w in range(len(Omega)):
        nameFig=nameFig+"_"+Omega[w]    
    
    nameFig=nameFig+"_nVess"
    for j in range(len(numVess)):
        nameFig=nameFig+"_"+numVess[j]
        
    nameFig=nameFig+"_dt"
    for j in range(len(dt)):
        nameFig=nameFig+"_"+dt[j]
        
    nameFig=nameFig+"_cell"
    for j in range(len(cell)):
        nameFig=nameFig+"_"+cell[j]
            
    
    nameFig=nameFig+".png"
    
    fig, axs = plt.subplots(1, 1, layout='constrained')
    
    for s in range(len(simulation)):
   
        for f in range(len(folder)):
        
            for c in range(len(cell)):
            
                for t in range(len(dt)):
        
                    for i in range(len(k)):
                        
                        for w in range(len(Omega)):
                        
                            for j in range(len(numVess)):
                                
                                if removeTime==0:
                                   a=0
                                    
                                else:
                                    if dt[t]=="0.001":
                                        a=int(removeTime*1e3)
                                        
                                    elif dt[t]=="0.0001":
                                        a=int(removeTime*1e4)
                                        
                                    elif dt[t]=="1e-05":
                                        a=int(removeTime*1e5)
                                
                                if (simulation[s]=="elastic"):
                                    
                                    for dx in range(len(DXmax)):
                                        solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        dat = np.loadtxt(solve3DP)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]/mmHg
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                    
                                elif (simulation[s]=="pBase"):
                                    for p in range(len(pBase)):
                                    
                                        solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"
                                        
                                        dat = np.loadtxt(solve3DP)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]/mmHg
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                    
                                elif (simulation[s]=="visco"):
                                    
                                    for dx in range(len(DXmax)):
                                    
                                        solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                
                                        dat = np.loadtxt(solve3DP)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]/mmHg
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                        
                                elif (simulation[s]=="dx"):
                                   for dx in range(len(DXmax)):
                                       solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                               
                                       dat = np.loadtxt(solve3DP)
                                       
                                       x = dat[a:, 0]
                                       y = dat[a:, 1]/mmHg
                                       
                                       axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                       axs.set_title('External Pressure pe')
                                       axs.set_xlabel("time [s]")
                                       axs.set_ylabel(r"External Pressure [mmHg]")
                                       fig.legend(loc="outside lower center")
                                       plt.show()
                                       
                                elif (simulation[s]=="viscodx"):
                                   for dx in range(len(DXmax)):
                                       solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                               
                                       dat = np.loadtxt(solve3DP)
                                       
                                       x = dat[a:, 0]
                                       y = dat[a:, 1]/mmHg
                                       
                                       axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                       axs.set_title('External Pressure pe')
                                       axs.set_xlabel("time [s]")
                                       axs.set_ylabel(r"External Pressure [mmHg]")
                                       fig.legend(loc="outside lower center")
                                       plt.show()
                                        
                                
                
    plt.savefig(nameFig)           
    

if plotak:
    #k=["1e+08","2e+08","3e+08"]
    k=["4e+08"]
    #Omega=["1","0.05","0.005","0.0005"]
    Omega=["1"]
    numVess=["1","2","3"]
    
    dt=["0.001"]
    cell=["0"]
    
    DXmax=["1"]
    
    network="Bifurcation" # Tree, Bifurcation
    simulation=["elastic"] # pBase, visco, dx, viscodx, elastic
    folder=["Solve3dk"] #or "MeanPressurek", Solve3dk
    
    pBase=[]
    
    removeTime=0
    
    a=int(removeTime*float(dt[t]))

    nameFig= "Plotarea "+"TimeFrom"+str(removeTime)+network 
    
    for i in range (len(simulation)):
        nameFig=nameFig+"_"+simulation[i]
        if simulation[i]=="pBase":
            for p in range(len(pBase)):
                nameFig=nameFig+"_"+pBase[p]
        if simulation[i]=="dx":
            for dx in range(len(DXmax)):
                nameFig=nameFig+"_"+DXmax[dx]
    
    nameFig=nameFig+"k_"
    for i in range (len(k)):
        nameFig=nameFig+"_"+k[i]
        
    nameFig=nameFig+"_Omega"    
    for w in range(len(Omega)):
        nameFig=nameFig+"_"+Omega[w]    
    
    nameFig=nameFig+"_nVess"
    for j in range(len(numVess)):
        nameFig=nameFig+"_"+numVess[j]
        
    nameFig=nameFig+"_dt"
    for j in range(len(dt)):
        nameFig=nameFig+"_"+dt[j]
        
    nameFig=nameFig+"_cell"
    for j in range(len(cell)):
        nameFig=nameFig+"_"+cell[j]
        
    nameFig=nameFig+".png"
    
    fig, axs = plt.subplots(1, 1, layout='constrained')
    
    
    for s in range(len(simulation)):
        
        for f in range(len(folder)):
        
            for c in range(len(cell)):
            
                for t in range(len(dt)):
        
                    for i in range(len(k)):
                        
                        for w in range(len(Omega)):
                        
                            for j in range(len(numVess)):
                                
                                if removeTime==0:
                                   a=0
                                    
                                else:
                                    if dt[t]=="0.001":
                                        a=int(removeTime*1e3)
                                        
                                    elif dt[t]=="0.0001":
                                        a=int(removeTime*1e4)
                                        
                                    elif dt[t]=="1e-05":
                                        a=int(removeTime*1e5)
                                
                                if (simulation[s]=="elastic"):
                                    
                                    for dx in range(len(DXmax)):
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        dat = np.loadtxt(solve3Da)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                
                                elif (simulation[s]=="pBase"):
                                    for p in range(len(pBase)):
                                    
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"
                                        
                                        dat = np.loadtxt(solve3Da)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                        
                                elif (simulation[s]=="visco"):
                                    
                                    for dx in range(len(DXmax)):
                                    
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                
                                        dat = np.loadtxt(solve3Da)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                        
                                elif (simulation[s]=="dx"):
                                    for dx in range(len(DXmax)):
                                        
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                
                                        dat = np.loadtxt(solve3Da)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                    
                                elif (simulation[s]=="viscodx"):
                                    for dx in range(len(DXmax)):
                                        
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                
                                        dat = np.loadtxt(solve3Da)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                    
                                    
                             
                
                
    plt.savefig(nameFig)  


if plotqk:
    #k=["1e+08","2e+08","3e+08"]
    k=["4e+08"]
    #Omega=["1","0.05","0.005","0.0005"]
    Omega=["1"]
    numVess=["1", "2","3"]
    
    dt=["0.001"]
    cell=["0"]
    
    network="Bifurcation" # Tree, Bifurcation
    simulation=["elastic"] # pBase, visco, dx, viscodx, elastic
    folder=["Solve3dk"] #or "MeanPressurek", Solve3dk
    
    pBase=[]
    DXmax=["1"]
    
    removeTime=0
    
    nameFig= "Plotq "+"TimeFrom"+str(removeTime)+network
    
    for i in range (len(simulation)):
        nameFig=nameFig+"_"+simulation[i]
        if simulation[i]=="pBase":
            for p in range(len(pBase)):
                nameFig=nameFig+"_"+pBase[p]
        if simulation[i]=="dx":
            for dx in range(len(DXmax)):
                nameFig=nameFig+"_"+DXmax[dx]
    
    nameFig=nameFig+"k_"
    for i in range (len(k)):
        nameFig=nameFig+"_"+k[i]
        
    nameFig=nameFig+"_Omega"    
    for w in range(len(Omega)):
        nameFig=nameFig+"_"+Omega[w]    
    
    nameFig=nameFig+"_nVess"
    for j in range(len(numVess)):
        nameFig=nameFig+"_"+numVess[j]
        
    nameFig=nameFig+"_dt"
    for j in range(len(dt)):
        nameFig=nameFig+"_"+dt[j]
        
    nameFig=nameFig+"_cell"
    for j in range(len(cell)):
        nameFig=nameFig+"_"+cell[j]
        
    nameFig=nameFig+".png"
    
    fig, axs = plt.subplots(1, 1, layout='constrained')
    
    for s in range(len(simulation)):
    
        for f in range(len(folder)):
        
            for c in range(len(cell)):
            
                for t in range(len(dt)):
                    
                    for i in range(len(k)):
                 
                        for w in range(len(Omega)):
                        
                            for j in range(len(numVess)):
                                
                                if removeTime==0:
                                   a=0
                                    
                                else:
                                    if dt[t]=="0.001":
                                        a=int(removeTime*1e3)
                                        
                                    elif dt[t]=="0.0001":
                                        a=int(removeTime*1e4)
                                        
                                    elif dt[t]=="1e-05":
                                        a=int(removeTime*1e5)
                                
                                if (simulation[s]=="elastic"):
                                    
                                    for dx in range(len(DXmax)):
                                        solve3Dq="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/qVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                    
                                        dat = np.loadtxt(solve3Dq)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]
                                        
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                        
                                elif (simulation[s]=="pBase"):
                                    for p in range(len(pBase)):
                                    
                                        solve3Dq="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/qVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"
                                        
                                        dat = np.loadtxt(solve3Dq)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                    
                                elif (simulation[s]=="visco"):
                                    
                                    for dx in range(len(DXmax)):
                                    
                                        solve3Dq="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/qVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                
                                        dat = np.loadtxt(solve3Dq)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                        
                                elif (simulation[s]=="dx"):
                                    for dx in range(len(DXmax)):
                                        solve3Dq="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/qVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                
                                        dat = np.loadtxt(solve3Dq)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                                        
                                elif (simulation[s]=="viscodx"):
                                    for dx in range(len(DXmax)):
                                        solve3Dq="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/qVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                
                                        dat = np.loadtxt(solve3Dq)
                                        
                                        x = dat[a:, 0]
                                        y = dat[a:, 1]
                                        
                                        axs.plot(x,y, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])
                                        axs.set_title('External Pressure pe')
                                        axs.set_xlabel("time [s]")
                                        axs.set_ylabel(r"External Pressure [mmHg]")
                                        fig.legend(loc="outside lower center")
                                        plt.show()
                    
                    
    plt.savefig(nameFig)                            

if plot4Media:
    
    #Define par to choose what to plot

    k="1e+09"
    Omega="1"
    numVess="2"
    dt="0.001"
    cell="0"
    
    network="Bifurcation/dt"
    simulation="elastic"
    folder=["Solve3dk", "MeanPressurek" ] #or "MeanPressure"
    
    # valori  per pressione costante 0
    constPe0="dati/"+network+dt+"/"+simulation+"/"+"ConstExtPressure0/PresEXTVess3D_1Cell"+cell+"_k_1e+08dt"+dt+"_omega_0.txt"
    constP0="dati/"+network+dt+"/"+simulation+"/"+"ConstExtPressure0/PresVess3D_1Cell"+cell+"_k_1e+08dt"+dt+"_omega_0.txt"
    consta0="dati/"+network+dt+"/"+simulation+"/"+"ConstExtPressure0/areaVess3D_1Cell"+cell+"_k_1e+08dt"+dt+"_omega_0.txt"
    constq0="dati/"+network+dt+"/"+simulation+"/"+"ConstExtPressure0/qVess3D_1Cell"+cell+"_k_1e+08dt"+dt+"_omega_0.txt"
    
    solve3DPe="dati/"+network+dt+"/"+simulation+"/"+folder[0]+k+"_Omega"+Omega+"/PresEXTVess3D_"+numVess+"Cell"+cell+"_k_"+k+"dt"+dt+"_omega_"+Omega+".txt"
    MeanPe="dati/"+network+dt+"/"+simulation+"/"+folder[1]+k+"_Omega"+Omega+"/MEDIAPresEXTVess3D_"+numVess+"Cell"+cell+"_k_"+k+"dt"+dt+"_omega_"+Omega+".txt"
    
    solve3DP= "dati/"+network+dt+"/"+simulation+"/"+folder[0]+k+"_Omega"+Omega+"/PresVess3D_"+numVess+"Cell"+cell+"_k_"+k+"dt"+dt+"_omega_"+Omega+".txt"
    MeanP="dati/"+network+dt+"/"+simulation+"/"+folder[1]+k+"_Omega"+Omega+"/MEDIAPresVess3D_"+numVess+"Cell"+cell+"_k_"+k+"dt"+dt+"_omega_"+Omega+".txt"
    
    solve3Da="dati/"+network+dt+"/"+simulation+"/"+folder[0]+k+"_Omega"+Omega+"/areaVess3D_"+numVess+"Cell"+cell+"_k_"+k+"dt"+dt+"_omega_"+Omega+".txt"
    Meana="dati/"+network+dt+"/"+simulation+"/"+folder[1]+k+"_Omega"+Omega+"/MEDIAareaVess3D_"+numVess+"Cell"+cell+"_k_"+k+"dt"+dt+"_omega_"+Omega+".txt"
    
    solve3Dq= "dati/"+network+dt+"/"+simulation+"/"+folder[0]+k+"_Omega"+Omega+"/qVess3D_"+numVess+"Cell"+cell+"_k_"+k+"dt"+dt+"_omega_"+Omega+".txt"
    Meanq="dati/"+network+dt+"/"+simulation+"/"+folder[1]+k+"_Omega"+Omega+"/MEDIAqVess3D_"+numVess+"Cell"+cell+"_k_"+k+"dt"+dt+"_omega_"+Omega+".txt"
    
    nameFig="Plot"+simulation+"k"+k+"_Omega"+Omega+"_Vess"+numVess+"_dt"+dt+"_cell"+cell+".png"
    title="Plot_k"+k+"_Omega"+Omega+"_Vess"+numVess+"_dt"+dt+"_cell"+cell
    #############################################
    pExt=[]
    
    dataPext0 = np.loadtxt(constPe0)
    pExt.append(dataPext0)
    
    dataPext1 = np.loadtxt(solve3DPe)
    pExt.append(dataPext1)
    
    dataPext2 = np.loadtxt(MeanPe)
    pExt.append(dataPext2)
    
    xpExt0 = pExt[0][:, 0]
    ypExt0 = pExt[0][:, 1]/mmHg
    xpExt1 = pExt[1][:, 0]
    ypExt1 = pExt[1][:, 1]/mmHg
    xpExt2 = pExt[2][:, 0]
    ypExt2 = pExt[2][:, 1]/mmHg
    
    
    P=[]
    
    dataP0 = np.loadtxt(constP0)
    P.append(dataP0)
    
    dataP1 = np.loadtxt(solve3DP)
    P.append(dataP1)
    
    dataP2 = np.loadtxt(MeanP)
    P.append(dataP2)
    
    xP0 = P[0][:, 0]
    yP0 = P[0][:, 1]/mmHg
    xP1 = P[1][:, 0]
    yP1 = P[1][:, 1]/mmHg
    xP2 = P[2][:, 0]
    yP2 = P[2][:, 1]/mmHg
    
    a=[]
    
    dataa0=np.loadtxt(consta0)
    a.append(dataa0)
    
    dataa1=np.loadtxt(solve3Da)
    a.append(dataa1)
    
    dataa2=np.loadtxt(Meana)
    a.append(dataa2)
    
    xa0 = a[0][:, 0]
    ya0 = a[0][:, 1]
    xa1 = a[1][:, 0]
    ya1 = a[1][:, 1]
    xa2 = a[2][:, 0]
    ya2 = a[2][:, 1]
    
    q=[]
    
    dataq0=np.loadtxt(constq0)
    q.append(dataq0)
    
    dataq1=np.loadtxt(solve3Dq)
    q.append(dataq1)
    
    dataq2=np.loadtxt(Meanq)
    q.append(dataq2)
    
    xq0 = q[0][:, 0]
    yq0 = q[0][:, 1]
    xq1 = q[1][:, 0]
    yq1 = q[1][:, 1]
    xq2 = q[2][:, 0]
    yq2 = q[2][:, 1]
    
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(wspace=0.41, hspace=0.53, top=0.91,bottom=0.19, left=0.12, right=0.95) #messi dopo aver aggiustato manualmente stampando la figura
    
    if parTitle==1:
        fig.suptitle(title)
    
    axs[0, 0].plot(xpExt0, ypExt0) #label="pExt=0")
    axs[0, 0].plot(xpExt1, ypExt1) #label="solve3Dk")
    axs[0, 0].plot(xpExt2, ypExt2) #label="pExt=MeanSimulation")
    axs[0, 0].set_title('External Pressure pe')
    #axs[0, 0].set_ylim(0,10)
    axs[0,0].set_xlabel("time [s]")
    axs[0,0].set_ylabel(r"External Pressure [mmHg]")
    #axs[0,0].legend()
    
    axs[0, 1].plot(xP0, yP0)#label="pExt=0" )
    axs[0, 1].plot(xP1, yP1) #label="solve3Dk")
    axs[0, 1].plot(xP2, yP2) #label="pExt=MeanSimulation" )
    axs[0, 1].set_title('Pressure p')
    axs[0,1].set_xlabel("time [s]")
    axs[0,1].set_ylabel(r"Pressure [mmHg]")
    #axs[0,1].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3)
    #axs[0,1].legend()
    
    
    axs[1, 0].plot(xa0, ya0 )# label="pExt=0")
    axs[1, 0].plot(xa1, ya1, )# label="solve3Dk")
    axs[1, 0].plot(xa2, ya2, )# label="pExt=MeanSimulation" )
    axs[1, 0].set_title(r'Area a')
    axs[1,0].set_xlabel("time [s]")
    axs[1,0].set_ylabel("Area [cm^2]")
    axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    #axs[1,0].legend()
    
    axs[1, 1].plot(xq0, yq0, label="pExt=0" )
    axs[1, 1].plot(xq1, yq1,  label="solve3Dk" )
    axs[1, 1].plot(xq2, yq2, label="pExt=MeanSimulation")
    axs[1, 1].set_title('Flow')
    axs[1,1].set_xlabel("time [s]")
    axs[1,1].set_ylabel("flow [ml/s]")
    #axs[1,1].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3)
        
     #loc=1, bbox_to_anchor=(-0.1, -0.15, 1, -0.15), fontsize=9, labelspacing=0.05, ncol=3)
    
    #plt.legend(loc='lower left', bbox_to_anchor= (0.0, -0.1), ncol=3)
    
    #fig.subplots_adjust(bottom=0.25)
    
    #fig.legend(bbox_to_anchor=(0, 0), loc="lower left",borderaxespad=1, ncol=3)
    #fig.legend(loc='lower center',bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=3)
    fig.legend(loc="center", bbox_to_anchor=(0.5, 0.05),ncol=3 )
    #fig.legend(loc=8, ncol=3)
    #fig.tight_layout(rect=[0, -0.15, 1, -0.15])
    #fig.tight_layout()
    #plt.savefig(nameFig, bbox_inches='tight')
    plt.savefig(nameFig)
    plt.show()
    
    
    
if plotPeConst4:
    
    
    
    #Define par to choose what to plot
    PmmHg=["30mmHg","40mmHg", "50mmHg"]
    Pdyncm=["39996.6","53328.8", "66661"]
    numVess=["1"]
    avsp=1#per decidere se fare plot 4 o area vs pressione
    
    if avsp==0:
        nameFig="PlotPeConst_"
        fig, axs = plt.subplots(2, 2)
        fig.subplots_adjust(wspace=0.41, hspace=0.53, top=0.91,bottom=0.19, left=0.12, right=0.95) #messi dopo aver aggiustato manualmente stampando la figura
        
        if parTitle==1:
            fig.suptitle(title)
    else:
        nameFig="AvsP_PlotPeConst_"
    
    for i in range(len(PmmHg)):
        nameFig=nameFig+"_"+PmmHg[i]
        
    for i in range(len(numVess)):
        nameFig=nameFig+"_numVess"+numVess[i]
    
    title=nameFig
    nameFig=nameFig+".png"
    
    
        
    removeTime=9
    
    
    for j in range(len(numVess)):
    
        for i in range(len(PmmHg)):
        
        
            Pe= "dati/Bifurcation/PeConst"+PmmHg[i]+"/PresEXTVess3D_"+numVess[j]+"Cell0dt0.001_pres_"+Pdyncm[i]+".txt"
            P= "dati/Bifurcation/PeConst"+PmmHg[i]+"/PresVess3D_"+numVess[j]+"Cell0dt0.001_pres_"+Pdyncm[i]+".txt" 
            a= "dati/Bifurcation/PeConst"+PmmHg[i]+"/areaVess3D_"+numVess[j]+"Cell0dt0.001_pres_"+Pdyncm[i]+".txt"
            q= "dati/Bifurcation/PeConst"+PmmHg[i]+"/qVess3D_"+numVess[j]+"Cell0dt0.001_pres_"+Pdyncm[i]+".txt"
            
        
            
            
            
            #############################################à
            
            dataPe = np.loadtxt(Pe)
            xPe = dataPe[:, 0]
            yPe = dataPe[:, 1]/mmHg
          
            dataP = np.loadtxt(P)
            xP = dataP[:, 0]
            yP = dataP[:, 1]/mmHg
            
            dataa = np.loadtxt(a)
            xa = dataa[:, 0]
            ya = dataa[:, 1]
          
            dataq = np.loadtxt(q)
            xq= dataq[:, 0]
            yq = dataq[:, 1]
          
            if avsp==0:
                axs[0, 0].plot(xPe, yPe) #label="pExt=0")
                axs[0, 0].set_title('External Pressure pe')
                axs[0,0].set_xlabel("time [s]")
                axs[0,0].set_ylabel(r"External Pressure [mmHg]")
                #axs[0,0].legend()
                
                axs[0, 1].plot(xP, yP)#label="pExt=0" )
                axs[0, 1].set_title('Pressure p')
                axs[0,1].set_xlabel("time [s]")
                axs[0,1].set_ylabel(r"Pressure [mmHg]")
                #axs[0,1].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3)
                #axs[0,1].legend()
                
                
                axs[1, 0].plot(xa, ya)# label="pExt=0")
                axs[1, 0].set_title(r'Area a')
                axs[1,0].set_xlabel("time [s]")
                axs[1,0].set_ylabel("Area [cm^2]")
                axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                #axs[1,0].legend()
                
                axs[1, 1].plot(xq, yq, label=PmmHg[i]+"_vess"+numVess[j] )
                axs[1, 1].set_title('Flow')
                axs[1,1].set_xlabel("time [s]")
                axs[1,1].set_ylabel("flow [ml/s]")
               
                fig.legend(loc="center", bbox_to_anchor=(0.5, 0.05),ncol=3 )
                #fig.legend(loc=8, ncol=3)
                #fig.tight_layout(rect=[0, -0.15, 1, -0.15])
                #fig.tight_layout()
                plt.show()
                #plt.savefig(nameFig, bbox_inches='tight')
            
            else:
                if removeTime==0:
                   
                    yP = dataP[:, 1]/mmHg
                    ya = dataa[:, 1]
                    
                else:
                   
                    a=int(removeTime*1e3)
                    
                    yP = dataP[a:, 1]/mmHg
                    ya = dataa[a:, 1]
                        
                   
                
                
                plt.plot(yP, ya, label=PmmHg[i]+"_vess"+numVess[j] )
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                plt.title("area vs pressure")
                plt.xlabel("Pressure [mmHg]")
                plt.ylabel("Area [cm^2]")
                plt.legend()
                plt.show()
                
            
    
    plt.savefig(nameFig)
        
if plot4:
    #k=["4e+08","5e+08", "6e+08"]
    #k=["1e+06", "1e+07", "1e+08"]
    #k=["3e+08","5e+08","7e+08","9e+08","1e+09","3e+09" ]
    #k=["0", "1e+07", "1e+08", "3e+08"]
    #k=["0", "1e+08", "3e+08"]
    k=["1e+08"]
    #Omega=["1","0.05","0.005","0.0005"]
    #Omega=["1","0.5","0.05","0.005","0.0005"]
    #Omega=["1","0.5","0.05","0.005"]
    Omega=["1"]
    #numVess=["1","2", "3"]
    numVess=["1"]
    #numVess=["1379"]  #[0, 389, 1375, 1, 252, 253, 353, 1376, 1377, 1378] +1 per indice vasi max tree
    dt=["0.1", "0.01", "0.001", "0.0001"]
    #dt=["0.0001"]
    #dt=["1e-05"]
    
    cell=["0"]
    #cell=["0", "1"]
    #cell=["0","1", "2"]
    #cell=["0","1", "2", "3"]
    
    network="Bifurcation" # Tree, Bifurcation
    simulation=["elastic"] # pBase, visco, dx, viscodx, elastic
    folder=["Solve3dk"] #or "MeanPressurek", Solve3dk
    
    pBase=["13332.2"] #["9332.54"]
    
    DXmax=["1"]
    
    removeTime=0
    
    nameFig= "PlotSolve3dk "+"TimeFrom"+str(removeTime)+network
    
    for i in range(len(simulation)):
        nameFig=nameFig+"_"+simulation[i]
        if simulation[i]=="pBase":
            for p in range(len(pBase)):
                nameFig=nameFig+"_"+pBase[p]
        if simulation[i]=="dx":
            for dx in range(len(DXmax)):
                nameFig=nameFig+"_"+DXmax[dx]
    
    nameFig=nameFig+"k_"
    for i in range(len(k)):
        nameFig=nameFig+"_"+k[i]
        
    nameFig=nameFig+"_Omega"    
    for w in range(len(Omega)):
        nameFig=nameFig+"_"+Omega[w]    
    
    nameFig=nameFig+"_nVess"
    for j in range(len(numVess)):
        nameFig=nameFig+"_"+numVess[j]
     
    nameFig=nameFig+"_dt"
    for j in range(len(dt)):
        nameFig=nameFig+"_"+dt[j]
        
    nameFig=nameFig+"_cell"
    for j in range(len(cell)):
        nameFig=nameFig+"_"+cell[j]
        
     
    title=nameFig
    nameFig=nameFig+".png"
    
    fig, axs = plt.subplots(2, 2, layout='constrained')
    #fig.subplots_adjust(wspace=0.41, hspace=0.53, top=0.91,bottom=0.19, left=0.12, right=0.95) #messi dopo aver aggiustato manualmente stampando la figura
    
    if parTitle==1:
        fig.suptitle(title)
    
    
    for s in range(len(simulation)):
    
        for f in range(len(folder)):
        
            for c in range(len(cell)):
            
                for t in range(len(dt)):
                    
                    for i in range(len(k)):
                
                        for w in range(len(Omega)):
                        
                            for j in range(len(numVess)):
                                
                                if removeTime==0:
                                   a=0
                                    
                                else:
                                    if dt[t]=="0.001":
                                        a=int(removeTime*1e3)
                                        
                                    elif dt[t]=="0.0001":
                                        a=int(removeTime*1e4)
                                        
                                    elif dt[t]=="1e-05":
                                        a=int(removeTime*1e5)
                                        

                                
                                if (simulation[s]=="elastic"):
                                    
                                    for dx in range(len(DXmax)):
                
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        solve3Dq="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/qVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        
                                        dataPe = np.loadtxt(solve3DPe)
                                        
                                        xPe = dataPe[a:, 0]
                                        yPe = dataPe[a:, 1]/mmHg
                                        
                                        dataP = np.loadtxt(solve3DP)
                                        xP = dataP[a:, 0]
                                        yP = dataP[a:, 1]/mmHg
                                       
                                        dataa = np.loadtxt(solve3Da)
                                        xa = dataa[a:, 0]
                                        ya = dataa[a:, 1]
                                     
                                        dataq = np.loadtxt(solve3Dq)
                                        xq= dataq[a:, 0]
                                        yq = dataq[a:, 1]
                                        
                                        #fig.suptitle('Sharing both axes')
                                        
                                        axs[0, 0].plot(xPe, yPe)#linestyle='--') #label="pExt=0")
                                        axs[0, 0].set_title('External Pressure pe')
                                        axs[0,0].set_xlabel("time [s]")
                                        axs[0,0].set_ylabel(r"External Pressure [mmHg]")
                                        #axs[0,0].legend()
                                    
                                        axs[0, 1].plot(xP, yP)#linestyle='--')#label="pExt=0" )
                                        axs[0, 1].set_title('Pressure p')
                                        axs[0,1].set_xlabel("time [s]")
                                        axs[0,1].set_ylabel(r"Pressure [mmHg]")
                                    #axs[0,1].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3)
                                    #axs[0,1].legend()
                                    
                                    
                                        axs[1, 0].plot(xa, ya)# linestyle='--')# label="pExt=0")
                                        axs[1, 0].set_title(r'Area a')
                                        axs[1,0].set_xlabel("time [s]")
                                        axs[1,0].set_ylabel("Area [cm^2]")
                                        axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                                    #axs[1,0].legend()
                                    
                                        axs[1, 1].plot(xq, yq, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w]+simulation[s]+"dt"+dt[t])#linestyle='--',
                                        axs[1, 1].set_title('Flow')
                                        axs[1,1].set_xlabel("time [s]")
                                        axs[1,1].set_ylabel("flow [ml/s]")
                                     
                                        #fig.legend(loc="center", bbox_to_anchor=(0.5, 0.05),ncol=3 )
                                        fig.legend(loc='outside lower center')
                                        
                                    #fig.legend(loc=8, ncol=3)
                                    #fig.tight_layout(rect=[0, -0.15, 1, -0.15])
                                    #fig.tight_layout()
                                        plt.show()
                                  
                                elif (simulation[s]=="pBase"):
                                    for p in range(len(pBase)):
                                    
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"
                                        solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"    
                                        solve3Dq="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/qVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"
                                    
                            
                                        dataPe = np.loadtxt(solve3DPe)
                                        
                                        xPe = dataPe[a:, 0]
                                        yPe = dataPe[a:, 1]/mmHg
                                        
                                        dataP = np.loadtxt(solve3DP)
                                        xP = dataP[a:, 0]
                                        yP = dataP[a:, 1]/mmHg
                                       
                                        dataa = np.loadtxt(solve3Da)
                                        xa = dataa[a:, 0]
                                        ya = dataa[a:, 1]
                                     
                                        dataq = np.loadtxt(solve3Dq)
                                        xq= dataq[a:, 0]
                                        yq = dataq[a:, 1]
                             
                                
                               
                                       #fig.suptitle('Sharing both axes')
                                       
                                        axs[0, 0].plot(xPe, yPe)#linestyle='--') #label="pExt=0")
                                        axs[0, 0].set_title('External Pressure pe')
                                        axs[0,0].set_xlabel("time [s]")
                                        axs[0,0].set_ylabel(r"External Pressure [mmHg]")
                                       #axs[0,0].legend()
                                       
                                        axs[0, 1].plot(xP, yP)#linestyle='--')#label="pExt=0" )
                                        axs[0, 1].set_title('Pressure p')
                                        axs[0,1].set_xlabel("time [s]")
                                        axs[0,1].set_ylabel(r"Pressure [mmHg]")
                                       #axs[0,1].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3)
                                       #axs[0,1].legend()
                                       
                                       
                                        axs[1, 0].plot(xa, ya)# linestyle='--')# label="pExt=0")
                                        axs[1, 0].set_title(r'Area a')
                                        axs[1,0].set_xlabel("time [s]")
                                        axs[1,0].set_ylabel("Area [cm^2]")
                                        axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                                       #axs[1,0].legend()
                                       
                                        axs[1, 1].plot(xq, yq, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])#linestyle='--',
                                        axs[1, 1].set_title('Flow')
                                        axs[1,1].set_xlabel("time [s]")
                                        axs[1,1].set_ylabel("flow [ml/s]")
                                        
                                        #fig.legend(loc="center", bbox_to_anchor=(0.5, 0.05),ncol=3 )
                                        fig.legend(loc='outside lower center')
                                       #fig.legend(loc=8, ncol=3)
                                       #fig.tight_layout(rect=[0, -0.15, 1, -0.15])
                                       #fig.tight_layout()
                                        plt.show()
                               #plt.savefig(nameFig, bbox_inches='tight')
                                elif (simulation[s]=="visco"):
                                   
                                   for dx in range(len(DXmax)):
                                       
                                       solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                       solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                       solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"    
                                       solve3Dq="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/qVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                       
                                       dataPe = np.loadtxt(solve3DPe)
                                       
                                       xPe = dataPe[a:, 0]
                                       yPe = dataPe[a:, 1]/mmHg
                                       
                                       dataP = np.loadtxt(solve3DP)
                                       xP = dataP[a:, 0]
                                       yP = dataP[a:, 1]/mmHg
                                      
                                       dataa = np.loadtxt(solve3Da)
                                       xa = dataa[a:, 0]
                                       ya = dataa[a:, 1]
                                    
                                       dataq = np.loadtxt(solve3Dq)
                                       xq= dataq[a:, 0]
                                       yq = dataq[a:, 1]
                            
                               
                              
                                      #fig.suptitle('Sharing both axes')
                                      
                                       axs[0, 0].plot(xPe, yPe)#linestyle='--') #label="pExt=0")
                                       axs[0, 0].set_title('External Pressure pe')
                                       axs[0,0].set_xlabel("time [s]")
                                       axs[0,0].set_ylabel(r"External Pressure [mmHg]")
                                      #axs[0,0].legend()
                                      
                                       axs[0, 1].plot(xP, yP)#linestyle='--')#label="pExt=0" )
                                       axs[0, 1].set_title('Pressure p')
                                       axs[0,1].set_xlabel("time [s]")
                                       axs[0,1].set_ylabel(r"Pressure [mmHg]")
                                      #axs[0,1].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3)
                                      #axs[0,1].legend()
                                      
                                      
                                       axs[1, 0].plot(xa, ya)# linestyle='--')# label="pExt=0")
                                       axs[1, 0].set_title(r'Area a')
                                       axs[1,0].set_xlabel("time [s]")
                                       axs[1,0].set_ylabel("Area [cm^2]")
                                       axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                                      #axs[1,0].legend()
                                      
                                       axs[1, 1].plot(xq, yq, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w]+simulation[s])#linestyle='--',
                                       axs[1, 1].set_title('Flow')
                                       axs[1,1].set_xlabel("time [s]")
                                       axs[1,1].set_ylabel("flow [ml/s]")
                                       
                                       #fig.legend(loc="center", bbox_to_anchor=(0.5, 0.05),ncol=3 )
                                       fig.legend(loc='outside lower center')
                                      #fig.legend(loc=8, ncol=3)
                                      #fig.tight_layout(rect=[0, -0.15, 1, -0.15])
                                      #fig.tight_layout()
                                       plt.show()
                                       
                                elif (simulation[s]=="dx"):
                                    for dx in range(len(DXmax)):
                                        
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"    
                                        solve3Dq="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/qVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        
                                        dataPe = np.loadtxt(solve3DPe)
                                        
                                        xPe = dataPe[a:, 0]
                                        yPe = dataPe[a:, 1]/mmHg
                                        
                                        dataP = np.loadtxt(solve3DP)
                                        xP = dataP[a:, 0]
                                        yP = dataP[a:, 1]/mmHg
                                       
                                        dataa = np.loadtxt(solve3Da)
                                        xa = dataa[a:, 0]
                                        ya = dataa[a:, 1]
                                     
                                        dataq = np.loadtxt(solve3Dq)
                                        xq= dataq[a:, 0]
                                        yq = dataq[a:, 1]
                             
                                
                               
                                       #fig.suptitle('Sharing both axes')
                                       
                                        axs[0, 0].plot(xPe, yPe)#linestyle='--') #label="pExt=0")
                                        axs[0, 0].set_title('External Pressure pe')
                                        axs[0,0].set_xlabel("time [s]")
                                        axs[0,0].set_ylabel(r"External Pressure [mmHg]")
                                       #axs[0,0].legend()
                                       
                                        axs[0, 1].plot(xP, yP)#linestyle='--')#label="pExt=0" )
                                        axs[0, 1].set_title('Pressure p')
                                        axs[0,1].set_xlabel("time [s]")
                                        axs[0,1].set_ylabel(r"Pressure [mmHg]")
                                       #axs[0,1].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3)
                                       #axs[0,1].legend()
                                       
                                       
                                        axs[1, 0].plot(xa, ya)# linestyle='--')# label="pExt=0")
                                        axs[1, 0].set_title(r'Area a')
                                        axs[1,0].set_xlabel("time [s]")
                                        axs[1,0].set_ylabel("Area [cm^2]")
                                        axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                                       #axs[1,0].legend()
                                       
                                        axs[1, 1].plot(xq, yq, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])#linestyle='--',
                                        axs[1, 1].set_title('Flow')
                                        axs[1,1].set_xlabel("time [s]")
                                        axs[1,1].set_ylabel("flow [ml/s]")
                                        
                                        #fig.legend(loc="center", bbox_to_anchor=(0.5, 0.05),ncol=3 )
                                        fig.legend(loc='outside lower center')
                                        #fig.legend(loc='outside right upper')
                                       #fig.legend(loc=8, ncol=3)
                                       #fig.tight_layout(rect=[0, -0.15, 1, -0.15])
                                       #fig.tight_layout()
                                        plt.show()
                                        
                                elif (simulation[s]=="viscodx"):
                                    for dx in range(len(DXmax)):
                                        
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                        solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"    
                                        solve3Dq="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/qVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                        
                                        dataPe = np.loadtxt(solve3DPe)
                                        
                                        xPe = dataPe[a:, 0]
                                        yPe = dataPe[a:, 1]/mmHg
                                        
                                        dataP = np.loadtxt(solve3DP)
                                        xP = dataP[a:, 0]
                                        yP = dataP[a:, 1]/mmHg
                                       
                                        dataa = np.loadtxt(solve3Da)
                                        xa = dataa[a:, 0]
                                        ya = dataa[a:, 1]
                                     
                                        dataq = np.loadtxt(solve3Dq)
                                        xq= dataq[a:, 0]
                                        yq = dataq[a:, 1]
                             
                                
                               
                                       #fig.suptitle('Sharing both axes')
                                       
                                        axs[0, 0].plot(xPe, yPe)#linestyle='--') #label="pExt=0")
                                        axs[0, 0].set_title('External Pressure pe')
                                        axs[0,0].set_xlabel("time [s]")
                                        axs[0,0].set_ylabel(r"External Pressure [mmHg]")
                                       #axs[0,0].legend()
                                       
                                        axs[0, 1].plot(xP, yP)#linestyle='--')#label="pExt=0" )
                                        axs[0, 1].set_title('Pressure p')
                                        axs[0,1].set_xlabel("time [s]")
                                        axs[0,1].set_ylabel(r"Pressure [mmHg]")
                                       #axs[0,1].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3)
                                       #axs[0,1].legend()
                                       
                                       
                                        axs[1, 0].plot(xa, ya)# linestyle='--')# label="pExt=0")
                                        axs[1, 0].set_title(r'Area a')
                                        axs[1,0].set_xlabel("time [s]")
                                        axs[1,0].set_ylabel("Area [cm^2]")
                                        axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                                       #axs[1,0].legend()
                                       
                                        axs[1, 1].plot(xq, yq, label="k"+k[i]+"numVess"+numVess[j]+"Omega"+Omega[w])#linestyle='--',
                                        axs[1, 1].set_title('Flow')
                                        axs[1,1].set_xlabel("time [s]")
                                        axs[1,1].set_ylabel("flow [ml/s]")
                                        
                                        #fig.legend(loc="center", bbox_to_anchor=(0.5, 0.05),ncol=3 )
                                        fig.legend(loc='outside lower center')
                                        #fig.legend(loc='outside right upper')
                                       #fig.legend(loc=8, ncol=3)
                                       #fig.tight_layout(rect=[0, -0.15, 1, -0.15])
                                       #fig.tight_layout()
                                        plt.show()
                                        
           
            
           
            
        plt.savefig(nameFig)
    
if plotAvsP: 
    #k=["1e+08", "2e+08", "3e+08"]
    #k=["3e+08","4e+08", "5e+08", "6e+08"]
    #k=["3e+08", "5e+08", "7e+08","9e+08","1e+09", "3e+09"]
    #k=["1e+09", "3e+09"]
    k=["1e+10"]
    #k=["1e+06","1e+07", "1e+08" ]
    #k=["4e+08", "5e+08", "6e+08","7e+08", "8e+08", "1e+09"]
    Omega=["1"]
    #Omega=["1","0.05","0.005","0.0005"]
    #Omega=["1","0.5","0.05","0.005"]
    #Omega=["0.5"]
    numVess=["1","2","3"]
    #numVess=["1"]
    #dt=["1e-05"]
    dt=["0.001"]
    #cell=["0", "1", "2", "3"]
    cell=["0"]
    #cell=["0","1"]
    #cell=["0", "1", "2"]
    
    network="Bifurcation" # Tree, Bifurcation
    simulation=["visco"] # pBase, visco, dx, viscodx, elastic
    folder=["Solve3dk"] #or "MeanPressurek", Solve3dk
    
    plotTransmural=0
    
    
   # DXmax=[ "0.1", "0.08", "0.06"]
    DXmax=["1"]
    pBase=[]
    nameFig= "AvsP "+network  
    
    removeTime=9 #per togliere i tempi da 0 a 1 (oppure poi vedi meglio cosa eliminare) cosi da togliere le cose non diritte nei grafici
    
    nameFig= "AvsP "+"RemoveTime"+str(removeTime)+network 
    
    for i in range(len(simulation)):
        nameFig=nameFig+"_"+simulation[i]
        if simulation[i]=="pBase":
            for p in range(len(pBase)):
                nameFig=nameFig+"_"+pBase[p]
        if simulation[i]=="dx":
            for dx in range(len(DXmax)):
                nameFig=nameFig+"_"+DXmax[dx]
    
    nameFig=nameFig+"k_"
    for i in range(len(k)):
        nameFig=nameFig+"_"+k[i]
        
    nameFig=nameFig+"_Omega"    
    for w in range(len(Omega)):
        nameFig=nameFig+"_"+Omega[w]    
    
    nameFig=nameFig+"_nVess"
    for j in range(len(numVess)):
        nameFig=nameFig+"_"+numVess[j]
     
    nameFig=nameFig+"_dt"
    for j in range(len(dt)):
        nameFig=nameFig+"_"+dt[j]
        
    nameFig=nameFig+"_cell"
    for j in range(len(cell)):
        nameFig=nameFig+"_"+cell[j]
        
     
    title=nameFig
    nameFig=nameFig+".png"
    
    
    
    
    for s in range(len(simulation)):
    
        for f in range(len(folder)):
        
            for c in range(len(cell)):
            
                for t in range(len(dt)):
                    
                    for i in range(len(k)):
                
                        for w in range(len(Omega)):
                        
                            for j in range(len(numVess)):
                                
                                if (simulation[s]=="elastic"):
                                    
                                    for dx in range(len(DXmax)):
        
                                        solve3DP= "dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                       
                                        dataP = np.loadtxt(solve3DP)
                                        dataPe = np.loadtxt(solve3DPe)
                                        dataa = np.loadtxt(solve3Da)
                                        
                                        if removeTime==0:
                                           
                                            yP = dataP[:, 1]/mmHg
                                            ype= dataPe[:,1]/mmHg
                                            ya = dataa[:, 1]
                                            
                                        else:
                                            if dt[t]=="0.001":
                                                a=int(removeTime*1e3)
                                                
                                                yP = dataP[a:, 1]/mmHg
                                                ype= dataPe[a:,1]/mmHg
                                                ya = dataa[a:, 1]
                                                
                                            elif dt[t]=="0.0001":
                                                a=int(removeTime*1e4)
                                                
                                                yP = dataP[a:, 1]/mmHg
                                                ype= dataPe[a:,1]/mmHg
                                                ya = dataa[a:, 1]
                                        
                                            elif dt[t]=="1e-05":
                                                a=int(removeTime*1e5)
                                                
                                                yP = dataP[a:, 1]/mmHg
                                                ype= dataPe[a:,1]/mmHg
                                                ya = dataa[a:, 1]
                                        
                                        ytr=yP-ype
                                        plt.plot(ya, yP, label= "k"+k[i]+"_dt"+dt[t]+"_w"+Omega[w]+"_nV"+numVess[j])
                                        
                                        if plotTransmural:
                                            plt.plot(ya, ytr, label="transmural pressure"+"_nV"+numVess[j])
                                            
                                        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                                        plt.title("area vs pressure")
                                        plt.ylabel("Pressure [mmHg]")
                                        plt.xlabel("Area [cm^2]")
                                        plt.legend()
                                        plt.show()
                                    
                                elif (simulation[s]=="pBase"):
                                    for p in range(len(pBase)):
                                    
                                        solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"_pBase"+pBase[p]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"_pBase"+pBase[p]+".txt"
                                
                                        dataP = np.loadtxt(solve3DP)
                                        dataPe = np.loadtxt(solve3DPe)
                                        dataa = np.loadtxt(solve3Da)
                                        
                                        if removeTime==0:
                                           
                                            yP = dataP[:, 1]/mmHg
                                            ype= dataPe[:,1]/mmHg
                                            ya = dataa[:, 1]
                                            
                                        else:
                                            if dt[t]=="0.001":
                                                a=int(removeTime*1e3)
                                                
                                                yP = dataP[a:, 1]/mmHg
                                                ype= dataPe[a:,1]/mmHg
                                                ya = dataa[a:, 1]
                                                
                                            elif dt[t]=="0.0001":
                                                a=int(removeTime*1e4)
                                                
                                                yP = dataP[a:, 1]/mmHg
                                                ype= dataPe[a:,1]/mmHg
                                                ya = dataa[a:, 1]
                                        
                                            elif dt[t]=="1e-05":
                                                a=int(removeTime*1e5)
                                                
                                                yP = dataP[a:, 1]/mmHg
                                                ype= dataPe[a:,1]/mmHg
                                                ya = dataa[a:, 1]
                                        
                                        ytr=yP-ype
                                        plt.plot(ya, yP, label= "k"+k[i]+"_dt"+dt[t]+"_w"+Omega[w]+"_nV"+numVess[j])
                                        
                                        if plotTransmural:
                                            plt.plot(ya, ytr, label="transmural pressure"+"_nV"+numVess[j])
                                            
                                        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                                        plt.title("area vs pressure")
                                        plt.ylabel("Pressure [mmHg]")
                                        plt.xlabel("Area [cm^2]")
                                        plt.legend()
                                        plt.show()
                                        
                                        
                                elif (simulation[s]=="visco"):
                                    
                                    for dx in range(len(DXmax)):
                                        
                                        solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                        solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                        solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                        
                                        dataP = np.loadtxt(solve3DP)
                                        dataPe= np.loadtxt(solve3DPe)
                                        dataa = np.loadtxt(solve3Da)
                                        
                                        if removeTime==0:
                                           
                                            yP = dataP[:, 1]/mmHg
                                            ype= dataPe[:,1]/mmHg
                                            ya = dataa[:, 1]
                                            
                                        else:
                                            if dt[t]=="0.001":
                                                a=int(removeTime*1e3)
                                                
                                                yP = dataP[a:, 1]/mmHg
                                                ype= dataPe[a:,1]/mmHg
                                                ya = dataa[a:, 1]
                                                
                                            elif dt[t]=="0.0001":
                                                a=int(removeTime*1e4)
                                                
                                                yP = dataP[a:, 1]/mmHg
                                                ype= dataPe[a:,1]/mmHg
                                                ya = dataa[a:, 1]
                                        
                                            elif dt[t]=="1e-05":
                                                a=int(removeTime*1e5)
                                                
                                                yP = dataP[a:, 1]/mmHg
                                                ype= dataPe[a:,1]/mmHg
                                                ya = dataa[a:, 1]
                                        
                                        ytr=yP-ype
                                        plt.plot(ya, yP, label= "k"+k[i]+"_dt"+dt[t]+"_w"+Omega[w]+"_nV"+numVess[j])
                                        
                                        if plotTransmural:
                                            plt.plot(ya, ytr, label="transmural pressure"+"_nV"+numVess[j])
                                            
                                        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                                        plt.title("area vs pressure")
                                        plt.ylabel("Pressure [mmHg]")
                                        plt.xlabel("Area [cm^2]")
                                        plt.legend()
                                        plt.show()
                                        
                                elif (simulation[s]=="dx"):
                                     for dx in range(len(DXmax)):
                                         
                                         solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                         solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                         solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef1"+".txt"
                                         
                                         dataP = np.loadtxt(solve3DP)
                                         dataPe=np.loadtxt(solve3DPe)
                                         dataa = np.loadtxt(solve3Da)
                                         
                                         if removeTime==0:
                                            
                                             yP = dataP[:, 1]/mmHg
                                             ype= dataPe[:,1]/mmHg
                                             ya = dataa[:, 1]
                                             
                                         else:
                                             if dt[t]=="0.001":
                                                 a=int(removeTime*1e3)
                                                 
                                                 yP = dataP[a:, 1]/mmHg
                                                 ype= dataPe[a:,1]/mmHg
                                                 ya = dataa[a:, 1]
                                                 
                                             elif dt[t]=="0.0001":
                                                 a=int(removeTime*1e4)
                                                 
                                                 yP = dataP[a:, 1]/mmHg
                                                 ype= dataPe[a:,1]/mmHg
                                                 ya = dataa[a:, 1]
                                         
                                             elif dt[t]=="1e-05":
                                                 a=int(removeTime*1e5)
                                                 
                                                 yP = dataP[a:, 1]/mmHg
                                                 ype= dataPe[a:,1]/mmHg
                                                 ya = dataa[a:, 1]
                                         
                                         ytr=yP-ype
                                         plt.plot(ya, yP, label= "k"+k[i]+"_dt"+dt[t]+"_w"+Omega[w]+"_nV"+numVess[j]+"cell"+cell[c]+"dx"+DXmax[dx])
                                         
                                         if plotTransmural:
                                             plt.plot(ya, ytr, label="transmural pressure"+"_nV"+numVess[j]+"cell"+cell[c])
                                             
                                         plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                                         plt.title("area vs pressure")
                                         plt.ylabel("Pressure [mmHg]")
                                         plt.xlabel("Area [cm^2]")
                                         plt.legend()
                                         plt.show()
                                         
                                elif (simulation[s]=="viscodx"):
                                     for dx in range(len(DXmax)):
                                         
                                         solve3DP="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                         solve3DPe="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/PresEXTVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                         solve3Da="dati/"+network+"/dt"+dt[t]+"/"+simulation[s]+DXmax[dx]+"/"+folder[f]+k[i]+"_Omega"+Omega[w]+"/areaVess3D_"+numVess[j]+"Cell"+cell[c]+"_k_"+k[i]+"dt"+dt[t]+"_omega_"+Omega[w]+"DXmax"+DXmax[dx]+"kmRef0"+".txt"
                                         
                                         dataP = np.loadtxt(solve3DP)
                                         dataPe=np.loadtxt(solve3DPe)
                                         dataa = np.loadtxt(solve3Da)
                                         
                                         if removeTime==0:
                                            
                                             yP = dataP[:, 1]/mmHg
                                             ype= dataPe[:,1]/mmHg
                                             ya = dataa[:, 1]
                                             
                                         else:
                                             if dt[t]=="0.001":
                                                 a=int(removeTime*1e3)
                                                 
                                                 yP = dataP[a:, 1]/mmHg
                                                 ype= dataPe[a:,1]/mmHg
                                                 ya = dataa[a:, 1]
                                                 
                                             elif dt[t]=="0.0001":
                                                 a=int(removeTime*1e4)
                                                 
                                                 yP = dataP[a:, 1]/mmHg
                                                 ype= dataPe[a:,1]/mmHg
                                                 ya = dataa[a:, 1]
                                         
                                             elif dt[t]=="1e-05":
                                                 a=int(removeTime*1e5)
                                                 
                                                 yP = dataP[a:, 1]/mmHg
                                                 ype= dataPe[a:,1]/mmHg
                                                 ya = dataa[a:, 1]
                                         
                                         ytr=yP-ype
                                         plt.plot(ya, yP, label= "k"+k[i]+"_dt"+dt[t]+"_w"+Omega[w]+"_nV"+numVess[j]+"cell"+cell[c]+"dx"+DXmax[dx])
                                         if plotTransmural:
                                             plt.plot(ya, ytr, label="transmural pressure"+"_nV"+numVess[j]+"cell"+cell[c])
                                             
                                         plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                                         plt.title("area vs pressure")
                                         plt.ylabel("Pressure [mmHg]")
                                         plt.xlabel("Area [cm^2]")
                                         plt.legend()
                                         plt.show()
                                         
                                        
    plt.savefig(nameFig)
    