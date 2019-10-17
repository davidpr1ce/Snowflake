#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:43:57 2019

@author: dprice
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns






files = ['2018-11.csv', '2018-12.csv', '2019-01.csv', '2019-02.csv', '2019-03.csv', \
         '2019-04.csv', '2019-05.csv', '2019-06.csv', '2019-07.csv', '2019-08.csv']

region = [-180,0,10,70] # US
#region =[-180,-65,180,75] # GLOBAL

#storage array for average values
Hs = np.zeros((180,360,10))
HC = np.zeros((180,360,10))
for i in range(len(files)):
    print(i)
    X, Y, H, Hall= get_PArr(files[i], altitude=20000, binsize=5., edr_peak=0.12, \
                            region=region)
    Hs[:,:,i] = H
    HC[:,:,i] = Hall
    
    

#mean isnt the way to go, since airlines get added this leads to anomolous effects
#in the pacific
#this now calculates the means over only the months where we have data
t = np.zeros((180,360))
for i in range(len(Hs[:,0])):
    for j in range(len(Hs[0,:])):
        t[i,j] = np.nanmean(Hs[i,j,:])

Hav = t


DiffA = np.zeros((180,360,10))
for i in range(len(files)):
    print(i)
    X, Y, H, Hall = get_PArr(files[i], altitude=20000, binsize=5., edr_peak=0.12, \
                             region=region)
    DA = (Hav - H)*-1.
    DA[np.isnan(DA)] = 0.    
    DiffA[:,:,i] = DA

  
for i in range(len(files)):
    i=0
    fig, axs = plt.subplots(3,1, figsize=(8,15))
    X, Y, H, Hall= get_PArr(files[i], altitude=20000, binsize=5., edr_peak=0.12, \
                            region=region)
    zer = plot_mat_BM(t, axs[0], title='Global mean ', ctitle='Percentage',\
                      vmin=0, vmax=np.round(np.nanmax(t)+.5),nlevels=21, cmap='Greens',
                      region=region)
    one = plot_mat_BM(H, axs[1], title=files[i], ctitle='Percentage ',cmap='Reds',\
                      vmin=0, vmax=np.round(np.nanmax(H)+.5), nlevels=21, \
                      region=region)
    two = plot_mat_BM(DiffA[:,:,i],axs[2], cmap='bwr', title='Difference from mean',\
                      ctitle='Percentage', vmin=-np.max(DiffA), vmax=np.max(DiffA), \
                      alpha=0.6, nlevels=21, region=region)
    
    plt.savefig("/home/dprice/Documents/Pngs/" + files[i][0:7] + '.pdf')


'''
fig, axis = plt.subplots(5,2, figsize=(16,25))
axis=axis.flatten()
for i in range(len(DiffA[0,0,:])):
    title = ('Seasonal difference from mean for:  ' + files[i][0:7])
    plot_mat_BM(DiffA[:,:,i], axis[i], cmap='seismic', ctitle='Percentage Difference', \
                title=title, vmin=-np.max(DiffA), vmax=np.max(DiffA), alpha=0.6, \
                nlevels=21, region=region)
  

fig, axt = plt.subplots(1,1,figsize=(12,6))
plot_mat_BM(Hav, axt, title='Mean Turbulence Occurence - All data', ctitle='Percentage', \
            nlevels=21, cmap='Reds', region=region, vmax=10.1)

'''