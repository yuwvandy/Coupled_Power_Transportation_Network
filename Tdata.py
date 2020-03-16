# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:35:35 2020

@author: wany105
"""

"""Transportation Data Input: Transportation Network Structure, Traffic Data
"""

Tadjl = [
    ("1", ["2", "12"]),
    ("2", ["3"]),
    ("3", ["4", "10"]),
    ("4", ["5"]),
    ("5", ["6", "8"]),
    ("6", []),
    ("7", ["6", "13"]),
    ("8", ["7"]),
    ("9", ["4", "8"]),
    ("10", ["9"]),
    ("11", ["2", "3", "10", "14", "15"]),
    ("12", ["2", "11", "15"]),
    ("13", []),
    ("14", ["7", "10", "13"]),
    ("15", ["14"]),
    ("16", ["12", "15"]),
]


# Capacity of each link (Conjugated to Graph with order)
# Here all the 19 links have the same capacity
capacity = [300, 200, 200, 200, 350, 400, 500, 250, 250, 300, 500, 550, 200, 400, 300, 300, 200, 300, 200,\
            200, 400, 300, 500, 600, 200, 500, 400, 350]

# Free travel time of each link (Conjugated to Graph with order)
free_time = [7, 9, 9, 12, 3, 9, 5, 13, 5, 9, 9, 10, 9, 6, 9, 8, 7, 14, 11,\
             5, 8, 10, 9, 12, 8, 11, 10, 10]


#Specify the type of each intersection
#0 - intersection with no signs, 1 - intersection with signals, 2 - intersection with stop signs
#originally 44 links
InterType = [1]*28
#Define whether the signals function well in this road
SigFun = [1]*28
Cycle = [30]*28
Green = [10]*28
t_service = [10]*28
hd = [2.5]*28

#Link Functionality: Change according to the severity of road damages by disruptions
function = [1]*28

# Origin-destination pairs
origins = ["1", "1", "16", "16"]
destinations = ["6", "13", "6", "13"]
# Generated ordered OD pairs:
# first ("5", "15"), second ("5", "17"), third ("6", "15")...


# Demand between each OD pair (Conjugated to the Cartesian 
# product of Origins and destinations with order)
demand = [660, 800, 800, 600]

color = 'blue'
name = 'TX-transportation'
lat, lon = [29.286, 29.313], [-94.8, -94.777]
nodenum, edgenum = 16, 28
O, D = ["1", "1", "16", "16"], ["6", "13", "6", "13"]

nodefile, edgefile, Type = 'TN.csv', 'TL.csv', 'local'

##Value setting for solving flow
accuracy = 1e-7
detail = True
precision = 4




