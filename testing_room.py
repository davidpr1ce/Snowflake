#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:04:44 2019

@author: dprice
"""
import numpy as np
import pandas as pd
import metpy.calc as mp
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

#airports in question
aplatlon = airport_grid('KDEN', 'KATL', factor=.5)
db = get_db('2019-08-01 00:00:00+00', '2019-08-31 23:59:59+00', latlongrid=aplatlon)
db = interpolate_db(db)



t1 = time_transform('2019-08-01 00:00:00+00')
#two ways of doing this...
days = [(t1 + i) for i in np.linspace(0, 2592000, 31)] 
weeks = [time_transform('2019-08-' + str(i).zfill(2) + ' 00:00:00+00') for i in np.arange(1, 31, 7)]
Alayers = np.linspace(30000,40000,11)

times = days  #choose days or weeks


#initalise
edr = 0.06
out = np.zeros((len(times), len(Alayers)))

for t in range(len(times)-1):
       print(t)
       for a in range(len(Alayers)-1):
              #select db region of interest
              dbt = db[(db['utc_timestamp'] > times[t]) & (db['utc_timestamp'] < times[t+1]) &\
                      (db['altitude'] > Alayers[a]) & (db['altitude'] < Alayers[a+1])]
              
              if len(dbt) > len(dbt)*0.01:
                     out[t,a] = (len(dbt[dbt['edr_peak_value'] > edr])/len(dbt))*100.
              else:
                     out[t,a] = np.nan
                     
              
          
              
              
              
              
              
              
