#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:50:40 2019

@author: dprice
"""

def plot_latlon_scatter(DataFrame, *,clusters=np.array([0]), latlongrid=[-180, -75, 180, 75]):
     
    from mpl_toolkits.basemap import Basemap
    import metpy.calc as mp
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    lats = DataFrame['latitude']
    lons = DataFrame['longitude']

    fig, axs = plt.subplots(1,1,figsize=(12,12))
    
    bm = Basemap(resolution='l', epsg=3857, ax=axs, \
                 llcrnrlon=latlongrid[0] ,\
                 llcrnrlat=latlongrid[1] ,\
                 urcrnrlon=latlongrid[2] ,\
                 urcrnrlat=latlongrid[3] )
    
    
    bm.drawcoastlines()
    bm.fillcontinents(alpha=0.3, color='green')
    bm.drawcountries()
    
    if len(clusters) >  1:
        clist = clusters
        clabel = 'Clusters'
    else:
        clist = list(DataFrame['edr_peak_value'])
        clabel = 'Peak_EDR_Value'
    
    scatterplot = bm.scatter(np.array(lons), np.array(lats), latlon=True, c=clist, \
                             marker='o', cmap='viridis', ax=axs)
    
    cb=plt.colorbar(scatterplot, fraction=0.02, pad=0.04, cmap='viridis')
    cb.set_label(clabel)
    
    
    
    return
  