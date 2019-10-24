#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:30:59 2019

@author: dprice
"""

import numpy as np
import pandas as pd
import metpy.calc as mp
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


ap_mc = ['KATL', 'KDEN', 'VHHH', 'LFPG','LSZH']   #common airports - some not NA ones included

airports = ['RJTT', 'KATL', 'KDEN', 'VHHH', 'LFPG', 'LSZH', 'KLAX','EGLL','KJFK']


start, end = '2019-08-01 00:00:00+00', '2019-08-31 23:59:00+00'


weeks = [('2019-08-' + str(i).zfill(2) + ' 00:00:00+00') for i in np.arange(1, 31, 7)]
months = [('2019-' + str(i).zfill(2) + '-01 00:00:00+00') for i in np.arange(4, 9, 1)]

for i in range(len(months)-1):
       t,c = investigate_airport('KDEN', months[i], months[i+1])


#some shit plots
#t = grid_and_plot_data(db, latlongrid=grid , resolution=.1)
#t = plot_latlon_scatter(db, latlongrid=grid)
       
       



       
       