# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 23:56:28 2020

@author: wany105
"""

"""The script contains basic function shared by all scripts in the same folder
"""
import numpy as np

def dist(node1, node2):
    """Calculate the distance between two vertices
    """
    return np.sqrt(np.sum((node1 - node2)**2))

def sortget(A, Num, flag = 0):
    """Get the first Num minimum/maximum element in A (list)
    Firstly Sort the A; Take the first Num element and search their index
    Input: the list A, the number of Num, flag indicate whether the maximum or minimum
    """
    B = sorted(A, reverse = flag)  
    Index = []
    for b in B:
        Index.append(A.index(b))
        if(len(Index) == Num):
            break
    
    return Index

def indexget(A, a):
    """Get the index of all elements equaling a specified value
    Input: the numpy array A, return the index value
    """
    Index = []
    for i in range(len(A)):
        if(A[i] == a):
            Index.append(i)
    
    return Index

def Normalize(X, Type):
    """Normalize an numpy array
    Input: X, Output: normalized X using specific methods
    """
    import numpy as np
    
    if(Type == 'max'):
        X = X/np.max(X)
        return X    
    
    if(Type == 'mean_sig'):
        mu = np.mean(X)
        sigma = np.std(X)
        X = (X-mu)/sigma    
        return X
    
def onSegment(n1xy, n2xy, n3xy):
    if((n2xy[0] <= max(n1xy[0], n3xy[0])) and (n2xy[0] >= min(n1xy[0], n3xy[0]))\
       and (n2xy[1] <= max(n1xy[1], n3xy[1])) and (n2xy[1] >= min(n1xy[1], n3xy[1]))):
        return True
    return False

def onSegment(n1xy, n2xy, n3xy): 
    if ((n2xy[0] <= max(n1xy[0], n3xy[0])) and (n2xy[0] >= min(n1xy[0], n3xy[0]))\
        and (n2xy[1] <= max(n1xy[1], n3xy[1])) and (n2xy[1] >= min(n1xy[1], n1xy[1]))): 
        return True
    return False
  
def orientation(n1xy, n2xy, n3xy):
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  
      
    val = (float(n2xy[1] - n1xy[1]) * (n3xy[0] - n2xy[0])) - (float(n2xy[0] - n1xy[0]) * (n3xy[1] - n2xy[1])) 
    if (val > 0): 
        # Clockwise orientation 
        return 1
    elif (val < 0): 
        # Counterclockwise orientation 
        return 2
    else: 
        # Colinear orientation 
        return 0
  
# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(n1xy, n2xy, n3xy, n4xy): 
    """Detect whether two line segements (node1, node2), (node3, node4) intersect with each
    Return 1 if so, return 0 if not
    """
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(n1xy, n2xy, n3xy) 
    o2 = orientation(n1xy, n2xy, n4xy) 
    o3 = orientation(n3xy, n4xy, n1xy) 
    o4 = orientation(n3xy, n4xy, n2xy) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return vectorangle(n1xy, n2xy, n3xy, n4xy)
  
    # Special Cases 
  
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(n1xy, n3xy, n2xy)): 
        return vectorangle(n1xy, n2xy, n3xy, n4xy)
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(n1xy, n4xy, n2xy)): 
        return vectorangle(n1xy, n2xy, n3xy, n4xy)
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(n3xy, n1xy, n4xy)): 
        return vectorangle(n1xy, n2xy, n3xy, n4xy)
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(n3xy, n2xy, n4xy)): 
        return vectorangle(n1xy, n2xy, n3xy, n4xy)
  
    # If none of the cases 
    return 0

def vectorangle(n1xy, n2xy, n3xy, n4xy):
    """Calculate the angle between two vectors: (n1xy, n2xy), (n3xy, n4xy)
    """
    Xvector1 = n2xy[0] - n1xy[0]
    Yvector1 = n2xy[1] - n1xy[1]
    
    Xvector2 = n4xy[0] - n3xy[0]
    Yvector2 = n4xy[1] - n3xy[1]
    
    Lvector1 = dist(np.array(n1xy), np.array(n2xy))
    Lvector2 = dist(np.array(n3xy), np.array(n4xy))
    
    cos = (Xvector2*Xvector1 + Yvector2*Yvector1)/(Lvector1*Lvector2)
    sin = np.sqrt(1 - cos**2)
    
    return sin

def Basemap(Type, lat, lon):
    """Geographical Map within certain locations.
    The location is given by some longitude and latitude interval
    """
    import os
    os.environ['PROJ_LIB'] = r"C:\Users\wany105\AppData\Local\Continuum\anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share"
    
    from mpl_toolkits.basemap import Basemap #Basemap package is used for creating geography map
    from matplotlib import pyplot as plt
    import numpy as np
    
    latinter = lat[1] - lat[0]
    loninter = lon[1] - lon[0]
    
    if(Type == 'local'):
        Base = Basemap(projection = 'merc', resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=0, llcrnrlon = lon[0], llcrnrlat = lat[0], urcrnrlon = lon[1], urcrnrlat = lat[1])
    elif(Type == 'whole'):
        Base = Basemap(resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=0, llcrnrlon = lon[0], llcrnrlat = lat[1], urcrnrlon = lon[1], urcrnrlat = lat[1])
    
    plt.figure(figsize = (20, 10))    
    Base.drawcoastlines()
    Base.drawcountries()
    Base.drawmapboundary()
    Base.drawparallels(np.arange(lat[0] - latinter/5, lat[1] + 2*latinter/5, latinter/5), labels=[1,0,0,1], fontsize = 20)
    Base.drawmeridians(np.arange(lon[0] - loninter/5, lon[1] + 2*loninter/5, loninter/5), labels=[1,1,0,1], fontsize = 20)
    
    return Base

def Unit_Length(array):
    """Unify the length of the performance list
    """
    temp1, temp2 = array.shape
    max_length = 0
    for i in range(temp1):
        for j in range(temp2):
            if(len(array[i, j]) > max_length):
                max_length = len(array[i, j])

    array2 = np.zeros([temp1, temp2, max_length])
    for i in range(temp1):
        for j in range(temp2):
            single_perform = array[i, j]
            for k in range(max_length):
                if(k < len(single_perform)):
                    array2[i, j, k] = single_perform[k]
                else:
                    array2[i, j, k] = single_perform[-1]
    
    return array2

def Link_Flow_Barchart(data1, data2):
    """Plot the Barchart of two type of data
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize = (8, 6))
    width = 0.3
    plt.bar(np.arange(len(data1)), data1, width=width, label = 'Without signal')
    plt.bar(np.arange(len(data2))+ width, data2, width=width, label = 'With signal')
    plt.xlabel('Link number')
    plt.ylabel('Link flow')
    plt.xticks(np.arange(0, len(data1), 1))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
#    plt.savefig("link_flow.png", dpi = 1500, bbox_inches='tight')
    plt.show()
    
def Link_Time_Barchart(data1, data2):
    """Plot the Barchart of two type of data
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize = (8, 6))
    width = 0.3
    plt.bar(np.arange(len(data1)), data1, width=width, label = 'Without signal')
    plt.bar(np.arange(len(data2))+ width, data2, width=width, label = 'With signal')
    plt.xlabel('Link number')
    plt.ylabel('Link cost\time')
    plt.xticks(np.arange(0, len(data1), 1))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
    plt.show()
                
def OD_Time_Barchart(data1, data2):
    """Plot the Barchart of two type of data
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    width = 0.3
    plt.bar(np.arange(len(data1)), data1, width=width, label = 'Without signal')
    plt.bar(np.arange(len(data2))+ width, data2, width=width, label = 'With signal')
    plt.xlabel('OD pair number')
    plt.ylabel('Paths between OD pair cost/time')
    plt.xticks(np.arange(0, len(data1), 1))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
#    plt.savefig("path_time.png", dpi = 1500, bbox_inches='tight')
    plt.show()


link_flow1 = [734.20, 725.80, 734.20, 387.01, 504.80, 858.81, 858.81, 0.00, 601.19, 792.56, 342.80, 471.80, 342.80, 814.60, 0.00, 157.62, 309.69, 491.37, 0.00, 0.00, 958.68, 436.50, 1050.95, 0.00, 607.44, 1167.12, 669.38, 730.62]
link_flow2 = [735.44, 724.56, 735.44, 438.98, 299.45, 835.40, 835.40, 0.00, 624.60, 770.13, 271.15, 396.43, 271.15, 667.57, 0.00, 0.00, 368.12, 554.36, 0.00, 0.00, 925.47, 411.20, 1123.57, 0.00, 629.87, 1199.09, 612.11, 787.89]

link_time1 = [44.666, 243.149, 254.166, 37.238, 4.947, 37.687, 11.528, 13.000, 30.081, 74.763, 9.298, 10.812, 20.651, 21.480, 9.000, 8.091, 13.036, 29.114, 11.000, 5.000, 47.595, 16.723, 35.350, 12.000, 110.110, 59.986, 21.763, 38.483]
link_time2 = [59.922, 256.549, 270.831, 68.776, 17.231, 49.685, 25.845, 23.000, 49.221, 82.627, 21.323, 23.567, 28.561, 27.982, 19.000, 18.033, 34.052, 53.485, 21.000, 15.000, 57.387, 30.294, 58.424, 22.000, 141.052, 80.577, 33.226, 63.519]

path_time1 = [385.286, 429.972, 163.903, 208.589]
path_time2 = [475.059, 508.478, 251.744, 285.158]

Link_Flow_Barchart(link_flow1, link_flow2)

Link_Time_Barchart(link_time1, link_time2)
OD_Time_Barchart(path_time1, path_time2)



    

    