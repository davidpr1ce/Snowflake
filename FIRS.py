#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:12:47 2019

@author: dprice
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
firs = pd.read_json('/home/dprice/Documents/JSON/firs.geo.json')
codes = np.array([firs['properties'][i]['ICAOCODE'] for i in range(len(firs))])

ts = firs['geometry'][217]
tp = firs['properties'][217]

coords = np.array(ts['coordinates'])
fig, axs = plt.subplots(1,1, figsize=(12,6))

latlongrid=[-110,25,-70,45]
# setup mercator map projection.
m0 = Basemap(resolution='l',epsg=3857, ax=axs, \
            llcrnrlon=latlongrid[0],llcrnrlat=latlongrid[1],urcrnrlon=latlongrid[2],urcrnrlat=latlongrid[3])
m0.drawcoastlines()
m0.fillcontinents(alpha=0.4, color='green')
m0.drawcountries()
m0.drawstates()


def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( list(xy), facecolor='red', alpha=0.4 )
    plt.gca().add_patch(poly)

draw_screen_poly(coords[0,:,1], coords[0,:,0], m0)


