#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:51:58 2019

@author: dprice
"""

#calculate the average head or tail wind over a flight


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt


#LFPG - paris
#KJFK - new york
#RJAA - tokyo
db = get_db('2019-07-01 12:00:00+00', '2019-08-07 12:00:00+00', latlongrid=US, Alayer=[0, 40000])
dbf = find_flights(db, 'LFPG', 'KJFK')
tafis = dbt['metadata_tafi'].unique()
for i in range(len(tafis)):
    flight = db[db['metadata_tafi']== tafis[i]]
    flighti = flight_info(flight, Plot=True, APlot=True, Afilter=[0,40000])
    
    break

#, latlongrid=[-36, 51, -24, 56]
#db = past_conditions('2019-08-01 12:00:00+00', 6.0, Grid=True, Alayers=[30000, 40000], resolution=2., Gradients='temperature')

#dbt = db[ (db['latitude'] >=51) & (db['latitude'] <= 56) & (db['longitude'] >= -36) & (db['longitude'] <= -24)]
#dbt = plot_latlon_scatter(dbt, latlongrid=[-80, 25, 10, 65])
#tafis = dbt['metadata_tafi'].unique()

#tafis = ['1bddfb95-25d4-42d9-bc96-c893c1341434','2782630b-0e18-41a5-ad5c-78f72e0adcc7']

#flight = flight_info('08', tafis[0], Plot=True, Afilter=[30000,40000], Velocity=245., APlot=True)
#flight = flight_info('08', tafis[1], Plot=True, Afilter=[30000,40000], Velocity=245., APlot=True)
