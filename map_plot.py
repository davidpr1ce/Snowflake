#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:40:20 2019

@author: dprice
"""

import pandas as pd
import matplotlib.pyplot as plt
import descartes
import geopandas as gpd
import geoplot as gplt
from shapely.geometry import Point, Polygon
import seaborn as sns

world_map_L1 =  \
 gpd.read_file('/home/dprice/Documents/ShapeFiles/GSHHS_shp/c/GSHHS_c_L1.shp')


db1 = pd.read_csv("~/Documents/CSV/2019-01.csv")
crs = {'init': 'epsg:4326'} #coordinate reference system for lat lon
geometry = [Point(xy) for xy in zip( db1['longitude'], db1['latitude'])]
geo_db = gpd.GeoDataFrame(db1, crs = crs, geometry=geometry)
#now have a new database with all the previous variables but lat and lon have
#been converted to a shapely Point variable!

#plotting games
'''
fig, ax = plt.subplots(figsize= (15,15))
world_map_L1.plot(ax = ax, alpha=0.4, color='grey')
geo_db[geo_db['edr_mean_value'] > 0.25].plot(ax=ax, markersize=2, color='red', marker='o')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('January 2019: Points with mean EDR > 0.25')
plt.grid(True)
plt.savefig("/home/dprice/Documents/Plots/test.pdf", format='pdf')
'''

'''
ax = gplt.polyplot(world_map_L1, alpha=0.4, color='grey', figsize=(15,15))
gplt.pointplot(geo_db[geo_db['edr_mean_value'] > 0.18], ax=ax, \
               hue='temperature', legend=True, \
               facecolor='lightgray')
ax.set_title('Test  Plot')
plt.savefig("/home/dprice/Documents/Plots/test2.pdf", format='pdf')
'''


ax = gplt.polyplot(world_map_L1, alpha=0.4, color='grey', figsize=(15,8))
#gplt.kdeplot(geo_db[geo_db['edr_mean_value'] > 0.05], ax=ax, \
#              facecolor='lightgray', cmap='Reds', shade=True, \
#              gridsize=100)
geo_samp = db1[db1['edr_peak_value']>0.12]
sns.kdeplot(geo_samp['longitude'], geo_samp['latitude'], ax=ax, \
            shade=True, shade_lowest=False, legend=True, gridsize=100, \
            cbar=True,n_levels=30 )
ax.set_title('Test  Plot')
plt.savefig("/home/dprice/Documents/Plots/test3.pdf", format='pdf')


'''
#vv basic correlation stuff
import scipy as sp
import numpy as np

geo_dbf = geo_db[geo_db['edr_mean_value'] > 0.06]
edr_m = geo_dbf['edr_mean_value'].to_numpy()
temp = geo_dbf['temperature'].to_numpy()
alt = geo_dbf['altitude'].to_numpy()
winds = geo_dbf['wind_speed'].to_numpy()

fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot(311)
ax1.xlabel='Temperature (C)'
ax1.ylabel='EDR Mean Value'
plt.scatter(temp,edr_m, marker='o')
ax2 = plt.subplot(312)
plt.scatter(alt,edr_m,marker='o')
ax3 = plt.subplot(313)
plt.scatter(winds,edr_m, marker='o')

'''


def produce_cont(index,path):
    '''
    Function to produce simple contour maps of monthly location of turbulence events.
    Index refers to the month in question where 0 ==2018-11, 1==2018-12 etc.
    Will save as pngs into path directory
    '''
    
    files = ['2018-11.csv', '2018-12.csv', '2019-01.csv', '2019-02.csv', '2019-03.csv', \
             '2019-04.csv', '2019-05.csv', '2019-06.csv', '2019-07.csv', '2019-08.csv']
     
    world_map_L1 =  \
    gpd.read_file('/home/dprice/Documents/ShapeFiles/GSHHS_shp/c/GSHHS_c_L1.shp')  

    pn = '/home/dprice/Documents/CSV/' + files[index]
    datab = pd.read_csv(pn)
    
    
    crs = {'init': 'epsg:4326'} #coordinate reference system for lat lon
    geometry = [Point(xy) for xy in zip( datab['longitude'], datab['latitude'])]
    geo_db = gpd.GeoDataFrame(datab, crs = crs, geometry=geometry)

    ax = gplt.polyplot(world_map_L1, alpha=0.4, color='grey', figsize=(15,15))
    gplt.kdeplot(geo_db[geo_db['edr_peak_value'] > 0.12], ax=ax, \
               facecolor='lightgray', cmap='Reds', shade=True, n_levels=30)
    ax.set_title(files[index][0:7])
    outd = path + files[index][0:7] + '.png'
    plt.savefig(outd, format='png')
    
    return


