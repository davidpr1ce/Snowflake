#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:09:12 2019

@author: dprice
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

files = ['2019-04-v3.csv', '2019-05-v3.csv', '2019-06-v3.csv', '2019-07-v3.csv', '2019-08-v3.csv']

for f in range(len(files)):
    print(f)
    db = pd.read_csv("~/Documents/CSVnew/" + files[f])
    
    #add TRIGGERED REPORT FLAG COLUMN
    tafis = db['metadata_tafi'].dropna().unique()
    flaga = np.zeros(len(db))
    for j in range(len(tafis)):
        #j=5161
        flight = db[db['metadata_tafi']==tafis[j]]
        
        if (len(flight)>1):
            times = np.array(flight['utc_timestamp'])
            times = np.round(times/10., decimals=0)*10 #rounding times to nearest 10 to remove some issues with selection
            edr = np.array(flight['edr_peak_value'])
            dt = np.ediff1d(times) #dt in 1 direction
            dt = np.append(dt, dt[-1])
            
            i = np.where(dt <= 120.)[0]      #sometimes events are spaced like 70 seconds apart or so - this captures them, could be potentially reduced
            p = np.where(np.roll(dt,1) <= 120.)[0] #trying to grab the last point in an event as well by shifting forward
            p = p[1:]         #fixing the case where the roll drops a false flag at the start of the array by removing zeros (if first element of flight is an event it is captured in i)
            
            k = np.unique(np.append(i,p))
            
            dbindex = np.array(flight.index)
            zeros = np.zeros(len(flight))
            zeros[k] = 1.
            flaga[dbindex] = zeros
            
    db['report_flag'] = pd.Series(flaga).values
        
    outname = files[f][0:7] + '-v4.csv'    
    db.to_csv("~/Documents/CSVnew/" + outname)
  
       
    
#DAL DESTINATION CODE
'''
tafis = db['metadata_tafi'].dropna().unique()


variables = ['utc_timestamp', 'altitude', 'longitude', 'latitude', 'temperature', 'wind_direction', \
             'wind_speed', 'edr_peak_value', 'edr_mean_value']

test = pd.DataFrame(columns=variables)

for i in range(len(tafis)):
    f= interpolate_flight(db[db['metadata_tafi'] == tafis[i]])
    test = pd.concat([test,f])
    if (i%np.round(len(tafis)/10.) == 0): print(np.round((i/len(tafis))*100.), ' %')




db = db[db['flight_airline'] != 'DAL']
tafis = db['metadata_tafi'].dropna().unique()

indexs=np.array([])
for i in range(len(tafis)):
    t = (db[db['metadata_tafi'] == tafis[i]].index)[0]
    indexs = np.append(indexs, t)

ap = pd.read_csv("~/Documents/CSV/airport-locations.csv")
db = pd.read_csv('~/Documents/CSVnew/full.csv')
iata_ap = db['flight_departure_aerodrome'].dropna().unique()
aps = np.array(ap['airport_code'])
aa = np.intersect1d(iata_ap, aps)
index = np.array([])
for i in range(len(aa)):
    t = np.where(aps == aa[i])[0]
    index = np.append(index,t[0])
apsn = ap.iloc[index,:]
apsn.to_csv('~/Documents/CSV/airport-locations-trimmed.csv')
'''
   
    
    