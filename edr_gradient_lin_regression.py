#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:01:55 2019

@author: dprice
"""
import numpy as np
import matplotlib.pyplot as plt

t1 = time_transform('2019-08-01 0:00:00+00')
times = np.linspace(0,604800, 8) + t1

edr = np.array([])
wg = np.array([])
tg = np.array([])

for i in range(len(times)-1):
    i=4
    ti=time_transform(times[i], Back=True)
    tf=time_transform(times[i+1], Back=True)
    windg, edro = gradients(ti, tf, key='wind_vector', Alayers=[30750,31250], grid=True, resolution=2., OutA=True)
    tempg, edro = gradients(ti, tf, key='temperature', Alayers=[30750,31250], grid=True, resolution=2., OutA=True)
    
    edr = np.append(edr, edro.flatten())
    wg = np.append(wg, windg.flatten())
    tg = np.append(tg, tempg.flatten())
    break



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


fig,axs = plt.subplots(1,2,figsize=(20,10))
axs[0].scatter(wg, edr, marker='o', color='blue' )
axs[0].set_xlabel('Mean Wind Vector Gradient')
axs[0].set_ylabel('Mean EDR Value')

axs[1].scatter(tg, edr, marker='o', color='red')
axs[1].set_xlabel('Mean Temperature Gradient')
axs[1].set_ylabel('Mean EDR Value')



