# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:35:35 2020

@author: wany105
"""

"""Transportation Data Input: Transportation Network Structure, Traffic Data
"""

Tadjl = [
    ("1", ["2", "12"]),
    ("2", ["1", "3", "11"]),
    ("3", ["2", "4", "10"]),
    ("4", ["3", "5"]),
    ("5", ["4", "6", "8"]),
    ("6", ["5", "7"]),
    ("7", ["6", "8", "13"]),
    ("8", ["9", "7"]),
    ("9", ["4", "10", "8"]),
    ("10", ["3", "11", "9", "14"]),
    ("11", ["2", "12", "10", "15"]),
    ("12", ["1", "11", "16"]),
    ("13", ["7", "14"]),
    ("14", ["10", "15", "13"]),
    ("15", ["11", "16", "14"]),
    ("16", ["12", "15"]),
]


# Capacity of each link (Conjugated to Graph with order)
# Here all the 19 links have the same capacity
capacity = [3600]*12 + [1800]*2 + [5400]*2 + [3600]*4 + [1800]*10 + [3600]*2 + [1800]*2 \
            + [5400]*2 + [3600]*2 + [1800]*6

# Free travel time of each link (Conjugated to Graph with order)
free_time = [48.28]*10 + [56.33]*2 + [48.28]*2 + [56.33]*2 + [48.28]*2 + [56.33]*2 \
            + [48.28]*10 + [56.33]*2 + [48.28]*2 + [56.33]*4 + [48.28]*6


#Specify the type of each intersection
#0 - intersection with no signs, 1 - intersection with signals, 2 - intersection with stop signs
#originally 44 links
InterType = [1]*44
#Define whether the signals function well in this road
SigFun = [1]*44
Cycle = [60]*44
Green = [20]*44
t_service = [10]*44
hd = [2.5]*44

#Link Functionality: Change according to the severity of road damages by disruptions
function = [1]*44

# Origin-destination pairs
origins = ["1"]
destinations = ["6"]
# Generated ordered OD pairs:
# first ("5", "15"), second ("5", "17"), third ("6", "15")...


# Demand between each OD pair (Conjugated to the Cartesian 
# product of Origins and destinations with order)
demand = [10000]

color = 'blue'
name = 'TX-transportation'
lat, lon = [29.286, 29.313], [-94.8, -94.777]
nodenum, edgenum = 16, 44
O, D = ["1"], ["6"]

nodefile, edgefile, Type = 'TN.csv', 'TL.csv', 'local'

##Value setting for solving flow
accuracy = 1e-7
detail = True
precision = 4




