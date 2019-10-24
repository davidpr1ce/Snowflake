#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:01:55 2019

@author: dprice
"""
import numpy as np
import matplotlib.pyplot as plt

t1 = time_transform('2019-08-01 0:00:00+00')
times = np.arange(0, 2592000, 43200) + t1

edr = np.array([])
wg = np.array([])
tg = np.array([])
aw = np.array([])
at = np.array([])
aplatlon = airport_grid('KJFK', 'EGLL', factor=.1)
aplatlon = [-60, 40, 0, 60]

'''
for i in range(len(times)-1):
    print(i)
    ti=time_transform(times[i], Back=True)
    tf=time_transform(times[i+1], Back=True)
    print(ti, tf)
    windg, edro, altsw = gradients(ti, tf, key='wind_vector', Alayers=np.linspace(30000,40000,11), grid=True, latlongrid=aplatlon, resolution=2., OutA=True)
    tempg, edro, altst = gradients(ti, tf, key='temperature', Alayers=np.linspace(30000,40000,11), grid=True, latlongrid=aplatlon, resolution=2., OutA=True)
    
    
    edr = np.append(edr, edro)
    wg = np.append(wg, windg)
    tg = np.append(tg, tempg)
    aw = np.append(aw, altsw)
    at = np.append(at, altst)

'''

from sklearn.linear_model import LinearRegression
ida = np.isfinite(wg) & np.isfinite(edr)
idb = np.isfinite(tg) & np.isfinite(edr)


wx=wg[ida].reshape((-1,1))
tx=tg[idb].reshape((-1,1))
wy=edr[ida]
ty=edr[idb]
modelw = LinearRegression().fit(wx,wy)
modelt = LinearRegression().fit(tx,ty)

print(modelw.score(wx,wy) , modelt.score(tx,ty))


fig, axs = plt.subplots(11, 1, figsize=(4,20))
Alayers=np.linspace(30000,40000,11)

i=0
for a in Alayers[1:]:
       index = np.where(at == a)
       axs[i].scatter(tg[index], edr[index])
       i+=1
       

'''
fig,axs = plt.subplots(1,2,figsize=(20,10))
a = axs[0].scatter(wg, edr, marker='o', c=aw, cmap='viridis', alpha=0.75 )
axs[0].set_xlabel('Mean Wind Vector Gradient')
axs[0].set_ylabel('Mean EDR Value')
fig.colorbar(a, ax=axs[0])

axs[1].scatter(tg, edr, marker='o', c=at, cmap='viridis', alpha=0.75)
axs[1].set_xlabel('Mean Temperature Gradient')
axs[1].set_ylabel('Mean EDR Value')
fig.colorbar(a, ax=axs[1], label='Altitude')
'''


