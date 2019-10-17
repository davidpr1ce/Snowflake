#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:21:25 2019

@author: dprice
"""


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

files = ['2018-11.csv', '2018-12.csv', '2019-01.csv', '2019-02.csv', '2019-03.csv', \
         '2019-04.csv', '2019-05.csv', '2019-06.csv', '2019-07.csv', '2019-08.csv']

#initialise array
out = np.zeros((10,6,2))

for i in range(10):
    
    #read in csv
    db = pd.read_csv('/home/dprice/Documents/CSV/'+files[i])
    
    #filter by airline
    #DAL = delta airlines
    #SWA = southwest airlines
    #CPA = cathay pacific
    #UAL = united airlines
    #AFR = air france
    #SWR = Swiss airlines
    #QTR = qatar airways
    #QFA = qantas airways
    #THY = turkish airlines
  
    #peak value series for each airline
    DAL = db[db['flight_airline']=='DAL']['edr_peak_value']   #DAL is delta I think.
    SWA = db[db['flight_airline']=='SWA']['edr_peak_value']
    CPA = db[db['flight_airline']=='CPA']['edr_peak_value']
    UAL = db[db['flight_airline']=='UAL']['edr_peak_value']
    AFR = db[db['flight_airline']=='AFR']['edr_peak_value']
    SWR = db[db['flight_airline']=='SWR']['edr_peak_value']

    
    #percentage of peak values above T1
    out[i,0,0] = (DAL[DAL > 0.18].count()/DAL.count())*100.
    out[i,1,0] = (SWA[SWA > 0.18].count()/SWA.count())*100.
    out[i,2,0] = (CPA[CPA > 0.18].count()/CPA.count())*100.
    out[i,3,0] = (UAL[UAL > 0.18].count()/UAL.count())*100.
    out[i,4,0] = (AFR[AFR > 0.18].count()/AFR.count())*100.
    out[i,5,0] = (SWR[SWR > 0.18].count()/SWR.count())*100.
    
    out[i,0,1] = DAL.count()
    out[i,1,1] = SWA.count()
    out[i,2,1] = CPA.count()
    out[i,3,1] = UAL.count()
    out[i,4,1] = AFR.count()
    out[i,5,1] = SWR.count()
    
    
 
fig, axs = plt.subplots(2,1,sharex=True, figsize=(10,8))
#fig.subplots_adjust(hspace=0)
  

labels=['2018-11','2018-12','2019-01','2019-02', '2019-03', '2019-04', '2019-05', \
        '2019-06', '2019-07', '2019-08']

plt.xticks(range(10), labels)
plt.legend(['DAL','SWA','CPA','UAL','AFR','SWR'])
axs[0].plot(range(10), out[:,0,0], marker='o', color='b')
axs[0].plot(range(10), out[:,1,0], marker='o', color='g')
axs[0].plot(range(10), out[:,2,0], marker='o', color='y')
axs[0].plot(range(10), out[:,3,0], marker='o', color='r')
axs[0].plot(range(10), out[:,4,0], marker='o', color='purple')
axs[0].plot(range(10), out[:,5,0], marker='o', color='black')
axs[0].legend(['DAL','SWA','CPA','UAL','AFR','SWR'])

carray = ['b', 'g', 'y', 'r' ,'purple', 'black']


wm = np.zeros(10)
uswm = np.zeros(10)
owm = np.zeros(10)
USairlines = [0,1,3]
other = [2,4,5]
for j in range(10):
        weights = out[j,:,1]/np.nansum(out[j,:,1])  #normalised weights
        wm[j] = np.nansum(weights*out[j,:,0])/np.nansum(weights)
        
        weightsus = out[j,USairlines,1]/np.nansum(out[j,USairlines,1])
        uswm[j] = np.nansum(weightsus*out[j,USairlines,0])/np.nansum(weightsus)
        
        weighto = out[j,other,1]/np.nansum(out[j,other,1])
        owm[j] = np.nansum(weighto*out[j,other,0])/np.nansum(weighto)
        

axs[1].plot(range(10), wm, marker='o', color='b')
axs[1].plot(range(10), uswm, marker='o', color='r')
axs[1].plot(range(10), owm, marker='o', color='y')
axs[1].legend(['All', 'US', 'Others'])


  
    

'''  

pr[np.where(pr == 0.)] = np.nan

fig= plt.figure(figsize=(10,10))
ax = plt.subplot(211)
labels=['2018-11','2018-12','2019-01','2019-02', '2019-03', '2019-04', '2019-05', \
        '2019-06', '2019-07', '2019-08']
plt.xticks([0,1,2,3,4,5,6,7,8,9],labels)
plt.plot(range(10), pr[0,:,0], marker='o', color='r')
plt.plot(range(10), pr[1,:,0], marker='o', color='b')
plt.plot(range(10), pr[2,:,0], marker='o', color='y')
plt.xlabel('Month')
plt.ylabel('Normalised likelihood of Turbulence report')
plt.legend(['TR3','TR2', 'TR1'])
plt.title('Seasonal dependence on likelihood of peak EDR rate reports')

ax=plt.subplot(212)
plt.xticks([0,1,2,3,4,5,6,7,8,9],labels)
plt.plot(range(10), pr[0,:,1], marker='o', color='r')
plt.plot(range(10), pr[1,:,1], marker='o', color='b')
plt.plot(range(10), pr[2,:,1], marker='o', color='y')
plt.xlabel('Month')
plt.ylabel('Normalised likelihood of Turbulence report')
plt.legend(['TR3','TR2', 'TR1'])
plt.title('Seasonal dependence on likelihood of mean EDR rate reports')
'''

   
    