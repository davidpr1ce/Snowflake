#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:55:24 2019

@author: dprice

Attempt to begin building an investigation into deriving a risk index for city pairs
Factors:
    - time of year
    - geographical/altitude hotspots
    - average wind/temp gradients
    - average deviation from flight plans
    - likelihood of sigmets
    - airline?
    - location of terminator?
    
Most of these will depend on time of year I suppose, which will be the main input variable to determine a risk index.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#aps = ['VHHH', 'EGLL']
aps = ['KDEN', 'KATL']
ap = pd.read_csv('~/Documents/CSV/airport-locations-trimmed.csv')

ap1lat = (ap[ap['airport_code']==aps[0]])['lat']
ap1lon = (ap[ap['airport_code']==aps[0]])['lon']  
ap2lat = (ap[ap['airport_code']==aps[1]])['lat']
ap2lon = (ap[ap['airport_code']==aps[1]])['lon']


#aplatlon = airport_grid(aps[0], aps[1], factor=.5)
#db = get_db('2019-08-01 00:00:00+00', '2019-08-02 23:59:59+00', Alayer=[30000,40000])
#db = interpolate_db(db)
#dbt, meanlons, meanlats = city_pair(db, aps[0], aps[1], Plot=True, latlongrid=aplatlon )  #3 pronged example
#t = investigate_airport('KDEN','2019-08-01 00:00:00+00', '2019-08-31 23:59:59+00', Ashells=np.arange(5000,36000,1000), edr=0.18, lshell=2000., R=1.5)


db = get_db('2019-08-01 00:00:00+00', '2019-08-07 23:59:59+00', latlongrid=US)
dbf = find_flights(db, aps[0], aps[1], Both=False)
tafis = dbf[dbf['edr_peak_value']>0.2]['metadata_tafi'].dropna().unique()
flights = dbf[(dbf['metadata_tafi'] == tafis[0])]
flight = flight_info(flights[flights['metadata_tafi'] == tafis[0]], Plot=False)


#dbi = flight_birdseye(flight, R=1., resolution=.25)



#flights surrounding - side by side and above/below


#side by side first-----------------------------------------------------------------------
#flight = flight_info(flights[flights['metadata_tafi'] == tafis[1]])
#interpolate lat and lon positions onto equally spaced grid

lons = np.array(flight['longitude'])
lons = np.insert(lons,0,ap1lon.iloc[0])
lons = np.append(lons, ap2lon.iloc[0])

lats = np.array(flight['latitude'])
lats = np.insert(lats,0,ap1lat.iloc[0])
lats = np.append(lats, ap2lat.iloc[0])

alts = np.array(flight['altitude'])
alts = np.insert(alts,0,0)
alts = np.append(alts, 0)

fl = flight_len(flight)
R = 1. #define radius of circle ( 1 ~ 60 nautical miles)
npoints = fl/(R/2.) #number of circles or central points of circle
xnew = [i for i in range(len(lats))] # define new x points



#interpolating onto circle grid space
lons_i = np.interp(np.linspace(xnew[0], xnew[-1], int(npoints)), range(len(lons)), lons, period=360.)
lats_i = np.interp(np.linspace(xnew[0], xnew[-1], int(npoints)), range(len(lats)), lats, period=180.)
alts_i = np.interp(np.linspace(xnew[0], xnew[-1], int(npoints)), range(len(alts)), alts)



plt.plot(flight['longitude'], flight['latitude'],  marker='o')


flight = flight[flight['report_flag'] != 1.]

plt.scatter(flight['longitude'], flight['latitude'], marker='D', color='red', zorder=5)

plt.scatter(lons_i, lats_i, marker='+', color='yellow', zorder=6)



#cross section next ----------------------------------------------------------------------

#dbi is database of corridor

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import scipy.ndimage.filters
from matplotlib.colors import ListedColormap
#draws and appends polygons leading each point in lat/lons with a corridor of width - gl figuring that out tbh but it works.
polygons = [ Polygon([((lons_i-1)[i], (lats_i-1)[i]), ((lons_i-1)[i+1], (lats_i-1)[i+1]), ((lons_i+1)[i+1], (lats_i+1)[i+1]), ((lons_i+1)[i], (lats_i+1)[i])]) \
              for i in range(len(lons_i)-1)]

Alayers = np.linspace(0, 40000, 41)

cross = np.zeros((len(polygons), len(Alayers)-1))
for poly in range(len(polygons)):
       polygon = polygons[poly]
       for A in range(len(Alayers)-1):
              dbc = dbi[(dbi['altitude'] > Alayers[A]) & (dbi['altitude'] < Alayers[A+1])]
              
              points = [ Point(np.array(dbc['longitude'])[i], np.array(dbc['latitude'])[i]) for i in range(len(dbc))]
              bools = [polygon.contains(i) for i in points]
              
              if (len(dbc[bools]['metadata_tafi'].unique()) > 1) | (sum(bools))>5:     #contribution from more than one flight or enough points             
                     cross[poly,A] = dbc[bools]['edr_peak_value'].mean()
              else:
                     cross[poly,A] = 0.0

      
#PLOTS------------------------------------------------------------------------------------


#cross-section in altitude
#setting up image
#distance to central points of each polygon + final point in flight which is on the edge of the last polygon
fig, axs = plt.subplots(1,1,figsize=(20,10))
                     
xd = np.array([np.sum(np.sqrt(np.ediff1d(lats_i)**2. + np.ediff1d(lons_i)**2.)[0:i]) for i in range(len(lats_i))])

xxx, yyy = np.meshgrid(xd[:-1]+np.ediff1d(xd)/2., Alayers[:-1]+np.ediff1d(Alayers)/2.)
zc = scipy.ndimage.filters.gaussian_filter(cross, .1)    
levelsc = np.linspace(np.min(zc), np.max(zc)) 
alphasc = np.array([i*1.2 for i in levelsc/np.max(levelsc)])
my_cmapc = ListedColormap(np.array([plt.cm.Reds(levelsc[i], alpha=alphasc[i]) for i in range(len(levelsc))]))

#plotting

axs.set_ylim(0,40000)
cp2 = axs.contour(xxx,yyy,zc.T, zorder=2, levels=levelsc, linewidths=1.0, cmap=my_cmapc)
cpf2 = axs.contourf(xxx,yyy, zc.T, zorder=1, levels=levelsc, cmap='Reds')
flp2 = axs.plot(xd, alts_i, color='green', zorder=4, marker='D')
axs.vlines(xd, 0, 40000, alpha=0.4, linestyle='--')
axs.hlines(Alayers, 0, xd.max(), alpha=0.4, linestyle='--')
#cb = axs[1].colorbar(cpf2, location='right', drawedges=True)




















'''
for i in range(5,9):
       db = pd.read_csv('~/Documents/CSVnew/2019-' + str(i).zfill(2) + '-v5.csv')
       dbi = interpolate_db(db)
       dbi.to_csv('~/Documents/ICSV/2019-' + str(i).zfill(2) + '-v5i.csv')
'''

