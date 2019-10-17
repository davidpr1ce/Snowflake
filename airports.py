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


#t, C = investigate_airport('EGLL', '2019-08-01 12:00:00+00', '2019-08-31 12:00:00+00', lshell=4000.)


for i in airports:
       t,c = investigate_airport(i, start, end)


#some shit plots
#t = grid_and_plot_data(db, latlongrid=grid , resolution=.1)
#t = plot_latlon_scatter(db, latlongrid=grid)