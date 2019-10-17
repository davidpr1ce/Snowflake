#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:28:17 2019

@author: dprice
"""
#tafis = ['1bddfb95-25d4-42d9-bc96-c893c1341434','2782630b-0e18-41a5-ad5c-78f72e0adcc7']
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#, latlongrid=[-36, 51, -24, 56]

db = past_conditions('2019-08-01 12:00:00+00', 8.0, Grid=False, resolution=1., Alayers=[33500, 35500], latlongrid=[-36, 51, -24, 56])
dbt = db[ (db['latitude'] >=51) & (db['latitude'] <= 56) & (db['longitude'] >= -36) & (db['longitude'] <= -24)]
#dbt = dbt[(dbt['utc_timestamp']> time_transform('2019-08-01 7:12:40+00')) & (dbt['utc_timestamp'] < time_transform('2019-08-01 10:18:00+00'))]

dbt = dbt[(dbt['edr_mean_value'] > 0.05)]



dbt = plot_latlon_scatter(dbt, latlongrid=[-80, 25, 10, 65])
tafis = dbt['metadata_tafi'].unique()
#print(len(tafis))
#print(dbt[['observation_time', 'altitude']])
#t1 = time_transform(dbt['utc_timestamp'].min(), Back=True)
#t2 = time_transform(dbt['utc_timestamp'].max(), Back=True)



#print(t1, t2)

'818782a8-8a9a-4825-9e52-d96ecb5cf2fe'
'da6e9721-6047-46ee-abed-446aa13975ea'

flight = flight_info('08', tafis[0], Plot=True, Afilter=[30000,40000], APlot=True)
flight = flight_info('08', tafis[-1], Plot=True, Afilter=[30000,40000], APlot=True)

