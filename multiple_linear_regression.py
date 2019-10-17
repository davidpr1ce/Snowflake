#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:13:51 2019

@author: dprice
"""

import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

db = pd.read_csv("~/Documents/CSV/2019-01.csv")

db.describe()
db = db[['temperature', 'wind_speed','altitude','edr_peak_value','edr_mean_value']]

#remove columns where there is a nan present in the above points
db = db.dropna()
db.describe()

YP = db['edr_peak_value']
YM = db['edr_mean_value']

X = db[['temperature']]


"""
X = sm.add_constant(X)
model = sm.OLS(YM,X).fit()
predictions = model.predict(X)
print_model = model.summary()
print(print_model)
"""

#i think its the bimodal stuff again, need to seperate into altitude shells


