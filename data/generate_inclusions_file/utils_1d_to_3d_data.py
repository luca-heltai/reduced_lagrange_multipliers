#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:51:00 2020

@author: caiazzo
"""

import numpy as np

'''
0: time; 
1: area[cm2]; 
2: flow [cm3/s]; 
3: theta (auxiliar variable for viscoelastic model, approximates dqdx) [cm2/s]; 
4 pressure [dyne/cm2; 
the rest is not relevant.
'''
# a vessel structure to read in Lucas input
class vessel1d:
    def __init__(self):
        # geometry of the vessel
        self.x0 = []
        self.x1 = []
        self.L = 1
        self.mid_points = []
        self.directions = []
        self.areas0 = []
        self.AAAREA = 0.
        self.IDs = []
        self.ID = 0
        
        # points where 1D quantities are computed
        self.pts = []

        # 1D variable for coupling
        self.pressure = 0.
        self.area = 0.
        self.flow = 0.

    

    def discretize(self,h,verbose=0):
        # fill the array p with 3d discretization points
        if h>self.L:
            
            self.pts.append(self.x0)
            self.pts.append(self.x1)
        
        else: 
            
            direction = (self.x1 - self.x0)/self.L
            length = 0
            pLast = self.x0
            
            while length <= self.L:
                if verbose>2:
                    print("new point:", pLast)
                self.pts.append(pLast)
              #  self.areas0.append(self.areas0)
                pNew = pLast + h*direction
                length = np.linalg.norm(pNew - self.x0)
                pLast = pNew
        
        self.set_midpoints()
       
            
            
    def set_midpoints(self):
        
        for i in range(0,len(self.pts)-1):
            
           mNew = (self.pts[i]+self.pts[i+1])/2
           dirNew = np.round(self.pts[i+1]-self.pts[i],3)
           areaNew = self.AAAREA
           self.mid_points.append(mNew)
           self.directions.append(dirNew)
           self.areas0.append(areaNew)
           self.IDs.append(self.ID)
            
        

            
        
            
# class for the network
class mesh1d:
    
    
    
    def __init__(self,vessels=None,nodes=None):
        
        # output times
        self.times = []
        
        if not nodes==None:
            self.set_nodes(nodes)
            
            if not vessels==None: 
                self.set_vessels(vessels)
        
            
     
    def get_nodes(self,nodes):
        nodeFile = np.genfromtxt(nodes) 
        if len(nodeFile[0])==3:
            N = nodeFile
        else:
            N = []
            for k in range(0,len(nodeFile)):
                N.append([nodeFile[k,2],nodeFile[k,3],nodeFile[k,4]])
        return N   
    
    def set_nodes(self,nodes):
        self.N = self.get_nodes(nodes)
        self.nN = len(self.N)
    
    
    def set_vessels(self,vessels):
        # list of vessels
        self.v = []
        V = np.genfromtxt(vessels)
        self.nV = len(V)
        
        for i in range(0,self.nV):
            # generate vessels
            newVessel = vessel1d()
            print(int(V[i,0]))
            newVessel.x0 = np.asarray([self.N[ int(V[i,0])][0],
                                       self.N[ int(V[i,0])][1],
                                       self.N[ int(V[i,0])][2]])
            
            newVessel.x1 = np.asarray([self.N[ int(V[i,1])][0],
                                       self.N[ int(V[i,1])][1],
                                       self.N[ int(V[i,1])][2]])
            
            newVessel.L = np.linalg.norm(newVessel.x1 - newVessel.x0)
            # newVessel.areas0 = V[i,3]
            newVessel.AAAREA = V[i,3]
            newVessel.ID = V[i,19]
            
            self.v.append(newVessel)
     
        
    def discretize(self,h3D):
        for i in range(0,self.nV):
            self.v[i].discretize(h3D)
    
    def set_values(self,basename,constant_pressure=None,constant_area=None):
        
       for j in range(0,self.nV):
                P = np.genfromtxt( basename + str(j+1) + '.txt')
                if j==0:
                    # times
                    self.times= P[:,0]
                   
                self.v[j].flow = []
                self.v[j].pressure = []
                self.v[j].area = []
                
                if constant_pressure==None:
                    for t in range(0,len(P[:,0])):
                        # pressure
                        self.v[j].pressure.append(P[t,5])
                        # areas
                        self.v[j].area.append(P[t,1])
                        # flow
                        self.v[j].flow.append(P[t,2])
                        
                else:
                    for t in range(0,len(P[:,0])):
                        # pressure
                        self.v[j].pressure.append(constant_pressure)
                        # areas
                        self.v[j].area.append(constant_area)
                        self.v[j].flow.append(0.)
      
    
    ###########################################################################
    # return list of all times
    def get_times(self):
        return self.times
    
    
    # return list of all points
    def get_points(self):
        P = np.vstack([np.array(v.mid_points) for v in self.v])
        return P
    
    # return list of all points
    def get_directions(self):
        P = np.vstack([np.array(v.directions) for v in self.v])
        return P
    
    
    # pressure matrix
    def get_flows(self):
        PMatrix = []
        for t in range(0,len(self.times)):
            
            Pcolumn = []
            for v in self.v:
                if not len(self.times) == len(v.flow):
                    print(t,len(self.times) , len(v.flow))
                for p in v.mid_points:
                    Pcolumn.append(v.flow[t])
            PMatrix.append(Pcolumn)
            
    # pressure matrix
    def get_pressures(self):
        PMatrix = []
        for t in range(0,len(self.times)):
            
            Pcolumn = []
            for v in self.v:
                if not len(self.times) == len(v.pressure):
                    print(t,len(self.times) , len(v.pressure))
                for p in v.mid_points:
                    Pcolumn.append(v.pressure[t])
            PMatrix.append(Pcolumn)
                    
        
        # P = np.hstack( 
        #     [
        #         np.vstack(
        #             [np.array(v.pressure[t]) for v in self.v for p in v.mid_points ]) for t in range(0,len(self.times))
        #     ]
        #     )
        return PMatrix
    
    # radii matrix
    def get_areas(self):
        P = np.hstack( [np.vstack(
                [np.array(v.area[t]) for v in self.v for p in v.mid_points ]) for t in range(0,len(self.times))])
        return P
    
    def get_areas_reference(self):
        #P = np.hstack( [np.vstack(
        #        [np.array(v.area0[t]) for v in self.v for p in v.mid_points ]) for t in range(0,len(self.times))])
        P = np.hstack([np.array(v.areas0) for v in self.v])
        return P
    
    def get_vessel_ID(self):
        P = np.hstack([np.array(v.IDs) for v in self.v])
        return P
    ###########################################################################    
    





