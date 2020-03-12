# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:48:58 2020

@author: wany105
"""
import numpy as np
from matplotlib import pyplot as plt
from graph import Graph
import ShareFunction as sf

class Power(Graph):
    def __init__(self, graph_dict= None, color = None, name = None, lat = None, lon = None, nodenum = None, edgenum = None, nodefile = None, edgefile = None, Type = None):
        Graph.__init__(self, graph_dict, color, name, lat, lon, nodenum, edgenum, nodefile, edgefile, Type)

        
    def performance_flow(self):
        """Calculate the power performance based on flow variation along the time during cacasding failure
        """
        import numpy as np
        
        flowsum = np.sum(self.flow)
        self.fperformance.append(min(1, np.sum(self.flowsequence[-1])/flowsum))
        
    def performance_topology(self):
        """Calculate power performance based on topology variation
        """
        
        
            
        
        
        