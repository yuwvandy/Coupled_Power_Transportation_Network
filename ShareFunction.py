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
    
  

    