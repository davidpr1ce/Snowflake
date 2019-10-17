#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:55:24 2019

@author: dprice

Attempt to begin building an investigation into deriving a risk index for city pairs
Factors:
    - time of year
    - geographical/altitude hotspots
    - average wind/temp gradients
    - average deviation from flight plans
    - likelihood of sigmets
    - airline?
    - location of terminator?
    
Most of these will depend on time of year I suppose, which will be the main input variable to determine a risk index.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

aplatlon = airport_grid('KDEN', 'KATL', factor=.5)
#aplatlon=US
db = get_db('2019-08-01 12:00:00+00', '2019-08-31 12:00:00+00', latlongrid=aplatlon, Alayer=[30000, 40000])
#mc = common_pairs(db,dist=True)
#db, meanlons, meanlats = mean_flightpath(db, mc.index[i][0:4], mc.index[i][4:8], Plot=True)
db = interpolate_db(db)

dbt, meanlons, meanlats = city_pair(db, 'KDEN', 'KATL', Plot=True, latlongrid=aplatlon )  #3 pronged example

#pt = get_Pdb(db, 0.06, resolution=.2, Interpolate=True)
'''


'''