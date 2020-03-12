# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:43:07 2020

@author: wany105
"""
import ShareFunction as sf
import numpy as np
from matplotlib import pyplot as plt

class system(object):
    """System object: network of networks
    """
    def __init__(self, name, networks, inters = None):
        self.name = name
        self.networks = networks
        self.inters = inters
        
        self.nodenum(networks)
        
    def nodenum(self, networks):
        """Sum of the vertex number over all networks
        """
        self.Nnum = 0
        for network in networks:
            self.Nnum += network.Nnum
        
    def Zlevel(self):
        """Assign each network a Z coordinate so that we can plot them in different level
        """
        
        self.Zlevel = {}
        for i in range(len(self.networks)):
            self.Zlevel[self.networks[i].name] = i*50        
            
        
    def Systemplot3d(self):
        """Plot the whole system in 3D dimension
        XY coordinates are exactly the same while Z coordinate of each network is shifted a little bit
        Interdependence is linked among different networks in different planes
        """
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig = plt.figure(figsize = (15, 10))
        ax = fig.add_subplot(111, projection = '3d')
        
        self.Zlevel()
        
        #Within networks
        for i in range(len(self.networks)):
            network = self.networks[i]
            Normflow = sf.Normalize(network.flow, Type = 'max')
            
            ##Plane Coordinates
            x = np.arange(250, 3200, 1)
            y = np.arange(250, 3200, 1)
            x, y = np.meshgrid(x, y)
            z = np.array([[self.Zlevel[network.name]]*len(x)]*len(y), dtype = float)
            
            X = network.Nx
            Y = network.Ny
        
            #Network node plot
            ax.scatter3D(X, Y, self.Zlevel[network.name], depthshade = False, zdir = 'z', marker = 's', color = network.c, label = network.name, s = 40)
            ax.plot_surface(x, y, z, linewidth=0, antialiased=False, alpha=0.05, color = network.c)
        
            #Network edge plot
            for j in range(len(network.linkf)):
                fnode = network.linkf[j] - 1
                tnode = network.linkt[j] - 1
                
                ax.plot([X[fnode], X[tnode]], [Y[fnode], Y[tnode]], [self.Zlevel[network.name], self.Zlevel[network.name]], network.c, lw = 4*Normflow[fnode, tnode])   
        
        
        #Among networks
        for i in range(len(self.inters)):
            
            interdependency = self.inters[i]
            Normflow = sf.Normalize(interdependency.flow, Type = 'max')
            
            network1 = interdependency.network1
            network2 = interdependency.network2
            
            for j in range(network1.Nnum):
                for k in range(network2.Nnum):
                    if(interdependency.adj[j, k] == 1):
                        fnodex, fnodey, fnodez = network1.Nx[j], network1.Ny[j], self.Zlevel[network1.name]
                        tnodex, tnodey, tnodez = network2.Nx[k], network2.Ny[k], self.Zlevel[network2.name]
                        ax.plot([fnodex, tnodex], [fnodey, tnodey], [fnodez, tnodez], interdependency.c, lw = 4*Normflow[j, k])
                    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(frameon = 0)
        
        
    def fail_simu(self, hurricane):
        """Simulate the failure scenario due to hurricane
        """
        import numpy as np
        
        Power = self.networks[0]
        self.networks[0].nfail = np.ones(Power.Nnum)
        for i in range(Power.Nnum):
            temp = np.random.rand()
            if(temp <= hurricane.failprob[i]):
                self.networks[0].nfail[i] = 0
        
        self.networks[0].nfail[0] = 1 ##The 1st node in power network is power station, which is assumed to be protected during hurricane
        
        ##Global Fail Index
        self.nfail = np.ones(self.Nnum)
        self.nfail[0:self.networks[0].Nnum] = self.networks[0].nfail
    
    def local_global_adj_flow(self):
        """Convert the local adjacent and flow matrix to global
        """
        self.global_flowadj()
        self.global_Amatrix()
        
    def global_flowadj(self):
        """Combine the flow matrix within a power network
        with the flow between the power and transportation network
        """
        import numpy as np
        
        self.flowadj = np.zeros([self.Nnum, self.Nnum])
        
        self.flowadj[0:self.networks[0].Nnum, 0:self.networks[0].Nnum] = self.networks[0].flow
        self.flowadj[0:self.networks[0].Nnum, self.networks[0].Nnum:self.Nnum] = self.inters[0].flow
        
    def global_Amatrix(self):
        """Combine the adjacent matrix of each network and each pair of interdependency
        """
        import numpy as np
        
        self.A = np.zeros([self.Nnum, self.Nnum])
        self.A[0:self.networks[0].Nnum, 0:self.networks[0].Nnum] = self.networks[0].A
        self.A[0:self.networks[0].Nnum, self.networks[0].Nnum:self.Nnum] = self.inters[0].adj
    
    
    def Cascading_failure(self, up_bound, low_bound):
        """Simulate the cascading failure based on flow redistribution
        """
        self.failsequence = []
        self.failsequence.append(self.nfail)
        
        self.flowsequence = []
        self.flowsequence.append(self.flowadj)
        
        self.networks[0].flowsequence = [self.networks[0].flow]
        self.inters[0].flowsequence = [self.inters[0].flow]
        
        self.networks[0].fperformance = [1]
        
#        self.flow_redistribution()
        
        while(1):
            self.flow_redistribution()
            self.fail_update(up_bound, low_bound)
            if((self.failsequence[-1] == self.failsequence[-2]).all()):
                break
        
        
    def flow_redistribution(self):
        """Redistribute the flow after the initial failure - cascading failure
        """
        import numpy as np
        
        self.flowadj2 = np.copy(self.flowsequence[-1])
        stack = []
        stack.append(0)
        while(stack != []):
            v = stack.pop()
            for i in range(self.Nnum):
                if(self.flowadj2[v, i] != 0):
                    stack.append(i)
            if(v == 0):
                flow_sum = np.sum(self.flowadj2[v, :])
            else:
                flow_sum = np.sum(self.flowadj2[:, v])
            
            self.flowadj2[v, :] = self.flowadj2[v, :]*self.failsequence[-1]
            ratio = self.flowadj2[v, :]/np.sum(self.flowadj2[v, :])
            ratio[np.isnan(ratio)] = 0
            self.flowadj2[v, :] = flow_sum*ratio
            
            ##Convert the updated system flow to power flow
            self.networks[0].flowsequence.append(self.flowadj2[0:self.networks[0].Nnum, 0:self.networks[0].Nnum])
            ##Convert the updated system flow to interdependent flow
            self.inters[0].flowsequence.append(self.flowadj2[0:self.networks[0].Nnum, self.networks[0].Nnum:self.Nnum])
            ##Calculate the performance
            self.networks[0].performance_flow()
        
        self.flowsequence.append(self.flowadj2)
        
    def fail_update(self, up_bound, low_bound):
        """Update the failure sceneria of vertices and links based on current flow
        """
        
        self.nfail2 = np.copy(self.failsequence[-1])
        flow_origin = self.flowsequence[0]
        
        for i in range(self.Nnum):
            if(i != 0):
                flowsum = np.sum(self.flowadj2[:, i])
                flowsum0 = np.sum(flow_origin[:, i])
                if(flowsum > up_bound*flowsum0 or flowsum < low_bound*flowsum0): #flow exceeding the capacity or less than the normal function threshold
                    self.nfail2[i] = 0
        
        self.failsequence.append(self.nfail2)
                
            
        
        
                
        
        
        
        
        
        
            
        
    

        
        
        
        
                    
    