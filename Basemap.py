#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:25:08 2019

@author: dprice
"""

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# create new figure, axes instances.
fig, axs = plt.subplots(1,1, figsize=(22,12))
# setup mercator map projection.
m0 = Basemap(resolution='l',epsg=3857, ax=axs, \
            llcrnrlon=-100.,llcrnrlat=5,urcrnrlon=20.,urcrnrlat=70)
m0.drawcoastlines()
m0.fillcontinents(alpha=0.4, color='green')
m0.drawcountries()

db = read_all(New=True)

xc, yc, cont = create2d_data(db, 'count', [-180, -75, 180, 85], resolution=2.)
xx, yy = np.meshgrid(xc,yc)

m0.contourf(yy,xx, np.log(cont), latlon=True, cmap='Reds', levels=50, ax=axs, alpha=0.4)
