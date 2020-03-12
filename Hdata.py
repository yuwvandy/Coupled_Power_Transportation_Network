# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:33:13 2020

@author: wany105
"""

"""Hurricane Data input: Hurricane Location, Hurricane Longitude, Hurricane Latitude, Hurricane Wind Pressure, Wind Speed
"""
Hnum = 15
Hurricane_name = ['H1863', 'H1871', 'H1895', 'H1899', 'H1900', 'H1915', 'H1938', 'H1947', 'H1959', 'H1974', 'H1983', 'H1989', 'H1995', 'H2003', 'H2008']
Data = ['Hurricane1863.csv', 'Hurricane1871.csv', 'Hurricane1895.csv', 'Hurricane1899.csv', 'Hurricane1900.csv', 'Hurricane1915.csv',\
        'Hurricane1938.csv', 'Hurricane1947.csv', 'Hurricane1959.csv', 'Hurricane1974.csv', 'Hurricane1983.csv', 'Hurricane1989.csv',\
        'Hurricane1995.csv', 'Hurricane2003.csv', 'Hurricane2008.csv']

color = ['red', 'lightsalmon', 'orange', 'darkkhaki', 'olivedrab', \
         'yellowgreen', 'skyblue', 'teal', 'darkorchid', 'plum', \
         'darkviolet', 'black' ,'grey' ,'navy', 'wheat']

Data_Location = r"C:\Users\wany105\Desktop\traffic and power\User-Equilibrium-Solution-master\User-Equilibrium-Solution-master\Hurricane\\"

def GeoBound(Hnum, Hurricane_name, Data, Data_Location):
    """Find the boundary of the trajectory of the hurricane (in longitude and latitude form)
    """
    import pandas as pd
    
    Latitude = []
    Longitude = []
    for i in range(Hnum):
        CSV = pd.read_csv(Data_Location + Data[i])
        lon = CSV['Longitude']
        lat = CSV['Latitude']
        
        Latitude.append([min(lat), max(lat)])
        Longitude.append([min(lon), max(lon)])
        
    return Latitude, Longitude

Latitude, Longitude = GeoBound(Hnum, Hurricane_name, Data, Data_Location)