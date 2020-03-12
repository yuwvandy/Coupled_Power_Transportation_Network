# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:28:02 2020

@author: wany105
"""
import ShareFunction as sf
import numpy as np
from matplotlib import pyplot as plt

class Hurricane(object):
    def __init__(self, name, network, Latitude, Longitude, color):
        self.name = name
        self.network = network
        self.lat = Latitude
        self.lon = Longitude
        self.c = color
    
    def fileimport(self, data, location):
        '''import information about the hurricane:
        Location, Longitude, Latitude, Wind Pressure, Wind Speed
        '''
        import pandas as pd
        
        CSV = pd.read_csv(location + data)
        
        self.Infoextract(CSV)
    
    def Infoextract(self, CSV):
        '''Extract hurricane information
        '''
        import numpy as np
        
        self.Nlon = np.array(CSV['Longitude'])
        self.Nlat = np.array(CSV['Latitude'])
        self.wp = np.array(CSV['Wind Pressure'])
        self.ws = np.array(CSV['Wind Speed'])
    
    def Basemap(self, Type):
        """Geographical Map within certain locations.
        The location is given by some longitude and latitude interval
        """
        import os
        os.environ['PROJ_LIB'] = r"C:\Users\wany105\AppData\Local\Continuum\anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share"
        
        from mpl_toolkits.basemap import Basemap #Basemap package is used for creating geography map
        from matplotlib import pyplot as plt
        import numpy as np
        
        latinter = self.lat[1] - self.lat[0]
        loninter = self.lon[1] - self.lon[0]
        
        if(Type == 'local'):
            self.Base = Basemap(projection = 'merc', resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=0, llcrnrlon = self.lon[0], llcrnrlat = self.lat[0], urcrnrlon = self.lon[1], urcrnrlat = self.lat[1])
        elif(Type == 'whole'):
            self.Base = Basemap(resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=0, llcrnrlon = self.lon[0], llcrnrlat = self.lat[1], urcrnrlon = self.lon[1], urcrnrlat = self.lat[1])
        
        plt.figure(figsize = (20, 10))    
        self.Base.drawcoastlines()
        self.Base.drawcountries()
        self.Base.drawmapboundary()
        self.Base.drawparallels(np.arange(self.lat[0] - latinter/5, self.lat[1] + latinter/5, latinter/5), labels=[1,0,0,1], fontsize = 10)
        self.Base.drawmeridians(np.arange(self.lon[0] - loninter/5, self.lon[1] + loninter/5, loninter/5), labels=[1,1,0,1], fontsize = 10)
        
    def verticexy(self, filename, filelocation, Type):
        '''load node coordinates information from CSV
        '''
        import numpy as np
        
        self.fileimport(filename, filelocation)
        self.Basemap(Type)
        self.Nx, self.Ny = self.Base(self.Nlon, self.Nlat)
    
    def trajectory_plot(self, townlon, townlat):
        '''Plot the trajectory of the hurricane
        '''
        import matplotlib.pyplot as plt
        
        townx, towny = self.Base(townlon, townlat)
        self.Base.scatter(townx, towny, marker = 'D', color = 'red', s = 200, label = 'Galveton, Texas')
        self.Base.plot(self.Nx, self.Ny, marker = 'D', color = 'm', label = 'Hurricane Trajectory')
        plt.legend()
    
    def ED_Rm(self):
        '''Calculate the eye diameter for each point along the hurricane trajectory
        '''
        import numpy as np
        
        self.ED = 46.29*np.exp(-0.0153*self.ws*1.852 + 0.0166*np.abs(self.Nlat))
        self.Rm = self.ED/2 + 8
        
    def NetworkXY(self):
        '''Convert the node XY in power network coordinate system to XY in hurricane system 
        '''
        import numpy as np
        
        self.netX, self.netY = self.Base(self.network.Nlon, self.network.Nlat)

    def Dist(self):
        '''Calculate the distance between vertices in power network and eye along the hurricane trajectory
        '''
        import numpy as np
        
        self.NetworkXY()
        self.Dist = np.zeros([len(self.netX), len(self.Nx)])
        for i in range(len(self.netX)):
            for j in range(len(self.Nx)):
                node1 = np.array([self.netX[i], self.netY[i]])
                node2 = np.array([self.Nx[j], self.Ny[j]])
                
                self.Dist[i, j] = sf.dist(node1, node2)/1000
      
    def v(self, a, b):
        '''Calculate the wind speed at specified location (x, y)
        a, b are parameters to calculate the wind speed.
        b is determined by the wind pressure
        a usually takes 0.5
        '''
        import numpy as np
        
        self.v = np.zeros_like(self.Dist)
        for i in range(len(self.netX)):
            for j in range(len(self.Nx)):
                temp = self.Rm[j]/self.Dist[i, j]
                self.v[i, j] = self.ws[j]*(temp**b*np.exp(1 - temp**b))**a
    
    def Failprob(self, mu, sigma, a, b):
        '''Calculate the fail probabiilty of each transmission towers
        mu, sigma: normal distribution parameters to calculate the failure probability
        '''
        from scipy.stats import norm
        import numpy as np
        
        self.ED_Rm()
        self.NetworkXY()
        self.Dist()
        self.v(a = 0.5, b = 2)
        
        self.failprob = np.zeros(len(self.netX))
        for i in range(len(self.netX)):
            temp = 0
            for j in range(len(self.Nx)):
                v = 1.287 * self.v[i, j] #Peak gust wind speed
                prob = norm.cdf(np.log(v/mu)/sigma)
                if(prob >= temp):
                    temp = prob
            self.failprob[i] = temp
            
        
        self.failprob *= 1 ##Lack of knowledge, need to be further researched on


        
        
        
        
                

    
        

        