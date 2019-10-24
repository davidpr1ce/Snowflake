#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:04:44 2019

@author: dprice
"""
import numpy as np
import pandas as pd
import metpy.calc as mp
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

#airports in question
#aplatlon = airport_grid('KDEN', 'KATL', factor=.5)
ap = pd.read_csv('~/Documents/CSV/airport-locations-trimmed.csv')
aps = ['KDEN', 'KATL']
latlongrid = airport_grid(aps[0], aps[1], factor=.5)
db = get_db('2019-08-01 00:00:00+00', '2019-08-02 23:59:59+00', latlongrid=latlongrid)
dbf = find_flights(db, aps[0], aps[1], Both=False)
tafis = dbf[dbf['edr_peak_value']>0.2]['metadata_tafi'].dropna().unique()
flights = dbf[(dbf['metadata_tafi'] == tafis[0])]
flight = flight_info(flights[flights['metadata_tafi'] == tafis[0]], Plot=False)


flighti = interpolate_flight(flight, interval=60.)
fla = flight_len(flighti, array=True) # array of distance from start
#re-interpolate onto 1deg distance grid
dstep = .5
npoints = int(fla[-1]/dstep)
xd = np.linspace(fla[0], fla[-1], int(fla[-1])) # 1deg distance array (ish)
loni = np.interp( xd, fla, flighti['longitude'])
lati = np.interp( xd, fla, flighti['latitude'])
alti = np.interp( xd, fla, flighti['altitude'])
utci = np.interp( xd, fla, flighti['utc_timestamp'])
   
fig, ax = plt.subplots(1,1,figsize=(10,10))
plt.plot(loni, lati, marker='o')
plt.plot(loni + 1, lati +1, marker='o')
plt.plot(loni - 1, lati -1, marker='o')

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#draws and appends polygons leading each point in lat/lons with a corridor of width - gl figuring that out tbh but it works.
width=1.
polygons = [[((loni-width)[i], (lati-width)[i]), ((loni-width)[i+1], (lati-width)[i+1]), ((loni+width)[i+1], (lati+width)[i+1]),\
              ((loni+width)[i], (lati+width)[i])] for i in range(len(loni)-1)]

for p in range(len(polygons)):
       xp, yp = Polygon(polygons[p]).exterior.xy
       plt.plot(xp,yp, color=plt.cm.viridis(p/len(polygons)), marker='D')



import scipy.ndimage.filters
from matplotlib.colors import ListedColormap
from geopandas import GeoSeries
#draws and appends polygons leading each point in lat/lons with a corridor of width - gl figuring that out tbh but it works.
Alayers = np.linspace(0, 40000, 41)

dbi = interpolate_db(db)


cross = np.zeros((len(polygons), len(Alayers)-1))
for poly in range(len(polygons)):
       polygon = polygons[poly]
       for A in range(len(Alayers)-1):
              dbc = dbi[(dbi['altitude'] > Alayers[A]) & (dbi['altitude'] < Alayers[A+1])]
              
              #points = [ Point(np.array(dbc['longitude'])[i], np.array(dbc['latitude'])[i]) for i in range(len(dbc))]
              points = GeoSeries(map(Point, zip(np.array(dbc['longitude']), np.array(dbc['latitude']))))
              
              bools = [Polygon(polygon).contains(i) for i in points]
              
              if (len(dbc[bools]['metadata_tafi'].unique()) > 1) | (sum(bools))>5:     #contribution from more than one flight or enough points             
                     cross[poly,A] = dbc[bools]['edr_peak_value'].mean()
              else:
                     cross[poly,A] = 0.0



def lonlat2dist(lons, lats, flight):
       '''
       attempt to convert lon,lat into distance along path for scatter plot in 2d
       
       '''
       #initial lon and lat for calculation
       loni, lati = ap[ap['airport_code'] == flight['flight_departure_aerodrome']]['lon'].iloc[0], \
                     ap[ap['airport_code'] == flight['flight_departure_aerodrome']]['lat'].iloc[0]
       
       
       
       
       
       return



fig, axs = plt.subplots(1,1, figsize=(20,10))
xx, yy = np.meshgrid((xd[:-1]+xd[1:])/2., (Alayers[:-1]+Alayers[1:])/2.)
z = scipy.ndimage.filters.gaussian_filter(cross, 1.)
levels=np.linspace(z.min(),z.max(), 10)

axs.contour(xx, yy ,z.T, zorder=2, levels=levels, linewidths=1.0, cmap='Reds')
axs.hlines(Alayers, xd.min(), xd.max(), linestyle='--', zorder=3,alpha=0.3)
axs.vlines(xd, 0, 40000, linestyle='--', zorder=3, alpha=0.3)
cf = axs.contourf(xx, yy, z.T, zorder=1, levels=levels, cmap='Reds')
cb = plt.colorbar(cf, ax=axs)
    